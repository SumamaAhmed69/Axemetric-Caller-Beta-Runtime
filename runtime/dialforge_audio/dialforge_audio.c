/**
 * Dialforge audio bridge for Baresip.
 *
 * BSD-3-Clause compatible module. It exposes one virtual audio source/player
 * named "dialforge" and exchanges 20 ms signed 16-bit mono PCM frames with
 * the Dialforge Python engine over loopback UDP only.
 */
#include <re.h>
#include <rem.h>
#include <baresip.h>
#include <re_atomic.h>
#include <re_thread.h>
#include <string.h>

#define DF_MAGIC0 'D'
#define DF_MAGIC1 'F'
#define DF_MAGIC2 'P'
#define DF_MAGIC3 'C'
#define DF_VERSION 1
#define DF_DIR_PHONE_TO_AI 1
#define DF_DIR_AI_TO_PHONE 2
#define DF_DIR_CLEAR 3
#define DF_HEADER_SIZE 24
#define DF_DEFAULT_SRATE 16000
#define DF_DEFAULT_PTIME 20
#define DF_MAX_PCM 3840
#define DF_QUEUE 64

struct df_src_st {
    struct ausrc_prm prm;
    ausrc_read_h *rh;
    void *arg;
};

struct df_play_st {
    struct auplay_prm prm;
    auplay_write_h *wh;
    void *arg;
};

struct df_frame {
    uint8_t data[DF_MAX_PCM];
    size_t len;
};

struct df_device {
    struct df_src_st *src;
    struct df_play_st *play;
    struct udp_sock *rx_sock;
    struct udp_sock *tx_sock;
    struct sa python_rx;
    thrd_t thread;
    RE_ATOMIC bool run;
    mtx_t lock;
    struct df_frame queue[DF_QUEUE];
    size_t q_head;
    size_t q_tail;
    size_t q_count;
    uint32_t generation;
    uint32_t sequence;
    uint32_t srate;
    uint32_t ptime;
    uint32_t rx_port;
    uint32_t tx_port;
};

static struct ausrc *g_ausrc;
static struct auplay *g_auplay;
static struct df_device g_dev;

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void wr16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}

static void wr32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void queue_clear(struct df_device *dev)
{
    dev->q_head = 0;
    dev->q_tail = 0;
    dev->q_count = 0;
}

static bool generation_newer(uint32_t candidate, uint32_t current)
{
    return candidate != current && (uint32_t)(candidate - current) < 0x80000000u;
}

static void queue_push(struct df_device *dev, const uint8_t *data, size_t len)
{
    if (!data || !len || len > DF_MAX_PCM)
        return;

    mtx_lock(&dev->lock);
    if (dev->q_count == DF_QUEUE) {
        dev->q_head = (dev->q_head + 1) % DF_QUEUE;
        --dev->q_count;
    }
    memcpy(dev->queue[dev->q_tail].data, data, len);
    dev->queue[dev->q_tail].len = len;
    dev->q_tail = (dev->q_tail + 1) % DF_QUEUE;
    ++dev->q_count;
    mtx_unlock(&dev->lock);
}

static size_t queue_pop(struct df_device *dev, uint8_t *out, size_t capacity)
{
    size_t len = 0;
    mtx_lock(&dev->lock);
    if (dev->q_count) {
        len = dev->queue[dev->q_head].len;
        if (len > capacity)
            len = capacity;
        memcpy(out, dev->queue[dev->q_head].data, len);
        dev->q_head = (dev->q_head + 1) % DF_QUEUE;
        --dev->q_count;
    }
    mtx_unlock(&dev->lock);
    return len;
}

static void udp_recv(const struct sa *src, struct mbuf *mb, void *arg)
{
    struct df_device *dev = arg;
    const uint8_t *p;
    size_t left;
    uint8_t direction;
    uint16_t header_size;
    uint32_t sequence;
    uint32_t srate;
    uint16_t channels;
    uint16_t samples;
    uint32_t generation;
    size_t expected;

    if (!src || !mb || !dev || !sa_is_loopback(src))
        return;

    left = mbuf_get_left(mb);
    if (left < DF_HEADER_SIZE)
        return;
    p = mbuf_buf(mb);
    if (p[0] != DF_MAGIC0 || p[1] != DF_MAGIC1 || p[2] != DF_MAGIC2 ||
        p[3] != DF_MAGIC3 || p[4] != DF_VERSION)
        return;

    direction = p[5];
    header_size = rd16(p + 6);
    sequence = rd32(p + 8);
    srate = rd32(p + 12);
    channels = rd16(p + 16);
    samples = rd16(p + 18);
    generation = rd32(p + 20);
    (void)sequence;

    if (header_size != DF_HEADER_SIZE || srate != dev->srate || channels != 1)
        return;
    expected = (size_t)samples * channels * 2;
    if (left != DF_HEADER_SIZE + expected || expected > DF_MAX_PCM)
        return;

    if (direction == DF_DIR_CLEAR) {
        mtx_lock(&dev->lock);
        if (generation == dev->generation || generation_newer(generation, dev->generation)) {
            dev->generation = generation;
            queue_clear(dev);
        }
        mtx_unlock(&dev->lock);
        return;
    }
    if (direction != DF_DIR_AI_TO_PHONE)
        return;

    mtx_lock(&dev->lock);
    if (generation_newer(generation, dev->generation)) {
        dev->generation = generation;
        queue_clear(dev);
    }
    if (generation != dev->generation) {
        mtx_unlock(&dev->lock);
        return;
    }
    mtx_unlock(&dev->lock);
    queue_push(dev, p + DF_HEADER_SIZE, expected);
}

static int send_phone_audio(struct df_device *dev, const void *pcm, size_t len)
{
    uint8_t hdr[DF_HEADER_SIZE];
    struct mbuf *mb;
    uint16_t samples;
    int err;

    if (!dev->tx_sock || !pcm || len > DF_MAX_PCM || len % 2)
        return EINVAL;
    samples = (uint16_t)(len / 2);
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = DF_MAGIC0; hdr[1] = DF_MAGIC1; hdr[2] = DF_MAGIC2; hdr[3] = DF_MAGIC3;
    hdr[4] = DF_VERSION;
    hdr[5] = DF_DIR_PHONE_TO_AI;
    wr16(hdr + 6, DF_HEADER_SIZE);
    wr32(hdr + 8, ++dev->sequence);
    wr32(hdr + 12, dev->srate);
    wr16(hdr + 16, 1);
    wr16(hdr + 18, samples);
    wr32(hdr + 20, dev->generation);

    mb = mbuf_alloc(DF_HEADER_SIZE + len);
    if (!mb)
        return ENOMEM;
    err = mbuf_write_mem(mb, hdr, sizeof(hdr));
    if (!err)
        err = mbuf_write_mem(mb, pcm, len);
    mb->pos = 0;
    if (!err)
        err = udp_send(dev->tx_sock, &dev->python_rx, mb);
    mem_deref(mb);
    return err;
}

static int device_thread(void *arg)
{
    struct df_device *dev = arg;
    const size_t sampc = (size_t)dev->srate * dev->ptime / 1000;
    const size_t bytes = sampc * 2;
    int16_t *remote = mem_zalloc(bytes, NULL);
    int16_t *ai = mem_zalloc(bytes, NULL);
    uint64_t ts = tmr_jiffies();

    if (!remote || !ai)
        goto out;

    info("dialforge_audio: bridge started %u Hz mono, %u ms\n", dev->srate, dev->ptime);
    while (re_atomic_rlx(&dev->run)) {
        uint64_t now;
        struct auframe af;
        size_t got;

        (void)sys_msleep(2);
        now = tmr_jiffies();
        if (ts > now)
            continue;

        memset(remote, 0, bytes);
        if (dev->play && dev->play->wh) {
            auframe_init(&af, AUFMT_S16LE, remote, sampc, dev->srate, 1);
            af.timestamp = ts * 1000;
            dev->play->wh(&af, dev->play->arg);
            (void)send_phone_audio(dev, remote, bytes);
        }

        memset(ai, 0, bytes);
        got = queue_pop(dev, (uint8_t *)ai, bytes);
        if (got < bytes)
            memset(((uint8_t *)ai) + got, 0, bytes - got);
        if (dev->src && dev->src->rh) {
            auframe_init(&af, AUFMT_S16LE, ai, sampc, dev->srate, 1);
            af.timestamp = ts * 1000;
            dev->src->rh(&af, dev->src->arg);
        }
        ts += dev->ptime;
    }

out:
    mem_deref(remote);
    mem_deref(ai);
    return 0;
}

static void maybe_start(struct df_device *dev)
{
    if (!dev->src || !dev->play || re_atomic_rlx(&dev->run))
        return;
    if (dev->src->prm.srate != dev->srate || dev->play->prm.srate != dev->srate ||
        dev->src->prm.ch != 1 || dev->play->prm.ch != 1 ||
        dev->src->prm.fmt != AUFMT_S16LE || dev->play->prm.fmt != AUFMT_S16LE) {
        warning("dialforge_audio: expected %u Hz mono s16\n", dev->srate);
        return;
    }
    re_atomic_rlx_set(&dev->run, true);
    if (thread_create_name(&dev->thread, "dialforge_audio", device_thread, dev))
        re_atomic_rlx_set(&dev->run, false);
}

static void stop_device(struct df_device *dev)
{
    if (re_atomic_rlx(&dev->run)) {
        re_atomic_rlx_set(&dev->run, false);
        thrd_join(dev->thread, NULL);
    }
    mtx_lock(&dev->lock);
    queue_clear(dev);
    mtx_unlock(&dev->lock);
}

static void src_destructor(void *arg)
{
    struct df_src_st *st = arg;
    stop_device(&g_dev);
    if (g_dev.src == st)
        g_dev.src = NULL;
}

static int src_alloc(struct ausrc_st **stp, const struct ausrc *as,
                     struct ausrc_prm *prm, const char *device,
                     ausrc_read_h *rh, ausrc_error_h *errh, void *arg)
{
    struct df_src_st *st;
    (void)as; (void)device; (void)errh;
    if (!stp || !prm || !rh)
        return EINVAL;
    st = mem_zalloc(sizeof(*st), src_destructor);
    if (!st)
        return ENOMEM;
    st->prm = *prm;
    st->rh = rh;
    st->arg = arg;
    g_dev.src = st;
    maybe_start(&g_dev);
    *stp = (struct ausrc_st *)st;
    return 0;
}

static void play_destructor(void *arg)
{
    struct df_play_st *st = arg;
    stop_device(&g_dev);
    if (g_dev.play == st)
        g_dev.play = NULL;
}

static int play_alloc(struct auplay_st **stp, const struct auplay *ap,
                      struct auplay_prm *prm, const char *device,
                      auplay_write_h *wh, void *arg)
{
    struct df_play_st *st;
    (void)ap; (void)device;
    if (!stp || !prm || !wh)
        return EINVAL;
    st = mem_zalloc(sizeof(*st), play_destructor);
    if (!st)
        return ENOMEM;
    st->prm = *prm;
    st->wh = wh;
    st->arg = arg;
    g_dev.play = st;
    maybe_start(&g_dev);
    *stp = (struct auplay_st *)st;
    return 0;
}

static int module_init(void)
{
    struct sa laddr;
    char host[64] = "127.0.0.1";
    uint32_t rx_port = 47500;
    uint32_t tx_port = 47501;
    uint32_t srate = DF_DEFAULT_SRATE;
    uint32_t ptime = DF_DEFAULT_PTIME;
    int err;

    memset(&g_dev, 0, sizeof(g_dev));
    if (mtx_init(&g_dev.lock, mtx_plain) != thrd_success)
        return ENOMEM;

    (void)conf_get_str(conf_cur(), "dialforge_audio_host", host, sizeof(host));
    (void)conf_get_u32(conf_cur(), "dialforge_audio_rx_port", &rx_port);
    (void)conf_get_u32(conf_cur(), "dialforge_audio_tx_port", &tx_port);
    (void)conf_get_u32(conf_cur(), "dialforge_audio_srate", &srate);
    (void)conf_get_u32(conf_cur(), "dialforge_audio_ptime", &ptime);
    if (str_cmp(host, "127.0.0.1") || srate < 8000 || srate > 48000 ||
        ptime < 10 || ptime > 40 || rx_port > 65535 || tx_port > 65535) {
        warning("dialforge_audio: unsafe or invalid bridge configuration\n");
        mtx_destroy(&g_dev.lock);
        return EINVAL;
    }
    g_dev.rx_port = rx_port;
    g_dev.tx_port = tx_port;
    g_dev.srate = srate;
    g_dev.ptime = ptime;
    g_dev.generation = 1;

    sa_set_str(&laddr, host, (uint16_t)tx_port);
    err = udp_listen(&g_dev.rx_sock, &laddr, udp_recv, &g_dev);
    if (err)
        goto fail;
    err = udp_listen(&g_dev.tx_sock, NULL, NULL, NULL);
    if (err)
        goto fail;
    sa_set_str(&g_dev.python_rx, host, (uint16_t)rx_port);

    err = ausrc_register(&g_ausrc, baresip_ausrcl(), "dialforge", src_alloc);
    if (err)
        goto fail;
    err = auplay_register(&g_auplay, baresip_auplayl(), "dialforge", play_alloc);
    if (err)
        goto fail;

    info("dialforge_audio: loopback bridge ready %s:%u/%u\n", host, rx_port, tx_port);
    return 0;

fail:
    g_ausrc = mem_deref(g_ausrc);
    g_auplay = mem_deref(g_auplay);
    g_dev.rx_sock = mem_deref(g_dev.rx_sock);
    g_dev.tx_sock = mem_deref(g_dev.tx_sock);
    mtx_destroy(&g_dev.lock);
    return err;
}

static int module_close(void)
{
    stop_device(&g_dev);
    g_ausrc = mem_deref(g_ausrc);
    g_auplay = mem_deref(g_auplay);
    g_dev.rx_sock = mem_deref(g_dev.rx_sock);
    g_dev.tx_sock = mem_deref(g_dev.tx_sock);
    mtx_destroy(&g_dev.lock);
    return 0;
}

EXPORT_SYM const struct mod_export DECL_EXPORTS(dialforge_audio) = {
    "dialforge_audio",
    "audio",
    module_init,
    module_close,
};
