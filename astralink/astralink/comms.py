"""Secure optical data link between Terminal 1 (client) and Terminal 2 (server).

Layering (as in the technical-approach diagram):
    app data -> TLS 1.3 (mutual cert authentication, real OpenSSL via ssl.MemoryBIO)
             -> packetization (MTU-sized frames with CRC32)
             -> lossy optical channel (loss/corruption driven by the *actual* pointing error)
             -> Selective-Repeat ARQ (cumulative ACK + SACK bitmap, per-frame timers)
             -> in-order reassembly -> TLS 1.3 decrypt -> app data
ARQ sits below TLS so TLS sees a reliable in-order byte stream (which TLS requires), while the
handshake itself also crosses the lossy link. Everything is stepped in simulation time."""
import datetime
import heapq
import hashlib
import ipaddress
import os
import ssl
import struct
import tempfile
import zlib
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID

DATA, ACK = 1, 2
HD = struct.Struct("!BI")
HA = struct.Struct("!BII")
CRC = struct.Struct("!I")


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Theseus FSOC"),
                      x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_pki(d):
    """Private CA + two terminal certificates (ECDSA P-256), generated fresh on every run."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(_name("FSOC Root CA")).issuer_name(_name("FSOC Root CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=2))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
          .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
          .sign(ca_key, hashes.SHA256()))
    paths = {"ca": os.path.join(d, "ca.pem")}
    open(paths["ca"], "wb").write(ca.public_bytes(serialization.Encoding.PEM))
    for cn in ("terminal1.fsoc.local", "terminal2.fsoc.local"):
        k = ec.generate_private_key(ec.SECP256R1())
        c = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(ca.subject).public_key(k.public_key())
             .serial_number(x509.random_serial_number())
             .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=2))
             .add_extension(x509.SubjectAlternativeName([x509.DNSName(cn)]), False)
             .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
             .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), True)
             .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]), False)
             .add_extension(x509.SubjectKeyIdentifier.from_public_key(k.public_key()), False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False)
             .sign(ca_key, hashes.SHA256()))
        cp, kp = os.path.join(d, cn + ".crt"), os.path.join(d, cn + ".key")
        open(cp, "wb").write(c.public_bytes(serialization.Encoding.PEM))
        open(kp, "wb").write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
        paths[cn] = (cp, kp)
    return paths


class Tls:
    def __init__(self, ctx, server, host=None):
        self.inc, self.out = ssl.MemoryBIO(), ssl.MemoryBIO()
        self.obj = ctx.wrap_bio(self.inc, self.out, server_side=server, server_hostname=host)
        self.done = False
        self.error = None

    def pump(self):
        if not self.done and self.error is None:
            try:
                self.obj.do_handshake()
                self.done = True
            except ssl.SSLWantReadError:
                pass
            except ssl.SSLError as e:
                self.error = str(e)

    def outbound(self):
        return self.out.read() if self.out.pending else b""

    def feed(self, b):
        self.inc.write(b)

    def write(self, b):
        self.obj.write(b)

    def read(self):
        buf = b""
        while True:
            try:
                x = self.obj.read(65536)
            except (ssl.SSLWantReadError, ssl.SSLZeroReturnError):
                break
            if not x:
                break
            buf += x
        return buf


class Channel:
    """Unidirectional optical channel: random fade-loss + bit corruption, fixed propagation delay."""

    def __init__(self, rng, delay, ber_frame):
        self.rng, self.delay, self.ber, self.q, self.n = rng, delay, ber_frame, [], 0
        self.sent = self.lost = self.corrupted = 0

    def send(self, frame, now, p_loss):
        self.sent += 1
        if self.rng.random() < p_loss:
            self.lost += 1
            return
        if self.rng.random() < self.ber:
            b = bytearray(frame)
            i = int(self.rng.integers(0, len(b)))
            b[i] ^= 1 << int(self.rng.integers(0, 8))
            frame = bytes(b)
            self.corrupted += 1
        self.n += 1
        heapq.heappush(self.q, (now + self.delay, self.n, frame))

    def deliver(self, now):
        out = []
        while self.q and self.q[0][0] <= now:
            out.append(heapq.heappop(self.q)[2])
        return out


class SRSender:
    def __init__(self, mtu, window, rto):
        self.mtu, self.win, self.rto = mtu, window, rto
        self.buf = bytearray()
        self.next = 0
        self.base = 0
        self.fl = {}          # seq -> [frame, last_tx_step, acked, nack_flag]
        self.tx = self.retx = self.nack_retx = 0

    def backlog(self):
        return len(self.buf) + sum(len(v[0]) for v in self.fl.values())

    def push(self, data):
        self.buf += data

    def on_ack(self, cum, sack):
        for q in [q for q in self.fl if q < cum]:
            del self.fl[q]
        self.base = max(self.base, cum)
        hi = -1
        for i in range(32):
            if sack >> i & 1:
                q = cum + 1 + i
                if q in self.fl:
                    self.fl[q][2] = True
                hi = max(hi, q)
        for q, v in self.fl.items():          # holes below the highest SACKed frame => implicit NACK
            if not v[2] and q < hi:
                v[3] = True

    def poll(self, step, budget):
        frames = []
        for q in sorted(self.fl):
            v = self.fl[q]
            if len(frames) >= budget:
                return frames
            if v[2]:
                continue
            if (v[3] and step - v[1] >= 4) or step - v[1] >= self.rto:
                if v[3]:
                    self.nack_retx += 1
                v[1], v[3] = step, False
                frames.append(v[0])
                self.retx += 1
        while len(frames) < budget and self.buf and self.next < self.base + self.win:
            payload = bytes(self.buf[:self.mtu])
            del self.buf[:self.mtu]
            h = HD.pack(DATA, self.next)
            fr = h + payload + CRC.pack(zlib.crc32(h + payload))
            self.fl[self.next] = [fr, step, False, False]
            self.next += 1
            frames.append(fr)
            self.tx += 1
        return frames


class SRReceiver:
    def __init__(self, window):
        self.win, self.exp, self.buf = window, 0, {}
        self.crc_fail = self.dups = 0

    def on_data(self, seq, payload):
        out = b""
        if seq < self.exp:
            self.dups += 1
        elif seq < self.exp + self.win:
            self.buf[seq] = payload
        while self.exp in self.buf:
            out += self.buf.pop(self.exp)
            self.exp += 1
        sack = 0
        for i in range(32):
            if self.exp + 1 + i in self.buf:
                sack |= 1 << i
        h = HA.pack(ACK, self.exp, sack)
        return out, h + CRC.pack(zlib.crc32(h))


class ArqEndpoint:
    def __init__(self, cfg):
        self.tx = SRSender(cfg.mtu, cfg.window, cfg.rto_steps)
        self.rx = SRReceiver(cfg.window)
        self.acks = []
        self.delivered = b""

    def on_frames(self, frames):
        for fr in frames:
            if len(fr) < 9 or zlib.crc32(fr[:-4]) != CRC.unpack(fr[-4:])[0]:
                self.rx.crc_fail += 1          # corrupted -> silently discarded, ARQ recovers
                continue
            if fr[0] == DATA:
                _, seq = HD.unpack(fr[:5])
                data, ack = self.rx.on_data(seq, fr[5:-4])
                self.delivered += data
                self.acks.append(ack)
            elif fr[0] == ACK:
                _, cum, sack = HA.unpack(fr[:9])
                self.tx.on_ack(cum, sack)

    def emit(self, step, budget):
        out, self.acks = self.acks[-2:], []     # keep only the freshest ACKs (they are cumulative)
        return out + self.tx.poll(step, budget)


class OpticalLink:
    def __init__(self, cfg, seed):
        self.c = cfg
        self.rng = np.random.default_rng(seed + 777)
        self.tmp = tempfile.TemporaryDirectory()
        pki = make_pki(self.tmp.name)
        srv = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        srv.minimum_version = ssl.TLSVersion.TLSv1_3
        srv.load_cert_chain(*pki["terminal2.fsoc.local"])
        srv.load_verify_locations(pki["ca"])
        srv.verify_mode = ssl.CERT_REQUIRED                        # terminal authentication (mutual)
        cli = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        cli.minimum_version = ssl.TLSVersion.TLSv1_3
        cli.load_cert_chain(*pki["terminal1.fsoc.local"])
        cli.load_verify_locations(pki["ca"])
        self.t1 = Tls(cli, False, "terminal2.fsoc.local")
        self.t2 = Tls(srv, True)
        self.e1, self.e2 = ArqEndpoint(cfg), ArqEndpoint(cfg)
        self.c12 = Channel(self.rng, cfg.prop_delay_steps, cfg.bit_error_prob)
        self.c21 = Channel(self.rng, cfg.prop_delay_steps, cfg.bit_error_prob)
        self.step_i = 0
        self.sent = bytearray()
        self.recv = bytearray()
        self.data_rng = np.random.default_rng(seed + 4242)
        self.t_established = None
        self.t_now = 0.0
        self.tls_version = None
        self.peer_subject = None

    def _micro(self, p_loss):
        c = self.c
        self.step_i += 1
        s = self.step_i
        for tls, ep in ((self.t1, self.e1), (self.t2, self.e2)):
            b = tls.outbound()
            if b:
                ep.tx.push(b)
        for f in self.e1.emit(s, c.frames_per_micro):
            self.c12.send(f, s, p_loss)
        for f in self.e2.emit(s, c.frames_per_micro):
            self.c21.send(f, s, p_loss)
        self.e2.on_frames(self.c12.deliver(s))
        self.e1.on_frames(self.c21.deliver(s))
        for tls, ep in ((self.t1, self.e1), (self.t2, self.e2)):
            if ep.delivered:
                tls.feed(ep.delivered)
                ep.delivered = b""
            tls.pump()
        if self.t1.done and self.t2.done and self.t_established is None:
            self.t_established = self.t_now
            self.tls_version = self.t1.obj.version()
            self.peer_subject = dict(x[0] for x in self.t2.obj.getpeercert().get("subject", ())) if self.t2.obj.getpeercert() else None
        r = self.t2.read() if self.t2.done else b""
        if r:
            self.recv += r
        self.t1.read() if self.t1.done else None

    def advance_frame(self, dt, p_loss):
        c = self.c
        n = int(round(dt / c.micro_dt_s))
        if self.t1.done and self.e1.tx.backlog() < c.app_backlog_limit:
            chunk = self.data_rng.bytes(c.app_bytes_per_frame)      # simulated payload / telemetry
            self.t1.write(chunk)
            self.sent += chunk
        for _ in range(n):
            self._micro(p_loss)
        self.t_now += dt

    def report(self, duration):
        got = bytes(self.recv)
        ok = hashlib.sha256(got).digest() == hashlib.sha256(bytes(self.sent[:len(got)])).digest()
        tx = self.e1.tx
        return dict(tls_version=self.tls_version, mutual_auth=self.peer_subject is not None or self.t_established is not None,
                    handshake_done_s=self.t_established, tls_error=self.t1.error or self.t2.error,
                    app_bytes_sent=len(self.sent), app_bytes_delivered=len(got), integrity_ok=bool(ok),
                    goodput_kBps=len(got) / max(duration, 1e-9) / 1000.0,
                    frames_first_tx=tx.tx, frames_retx=tx.retx,
                    retx_ratio=tx.retx / max(tx.tx, 1), channel_loss_frac=self.c12.lost / max(self.c12.sent, 1),
                    crc_discards=self.e2.rx.crc_fail)
