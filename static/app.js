// =====================================================================
// Penjaga Tenang Lab - logika halaman (hanya mode Pantau)
// Mikrofon (AudioWorklet) -> kirim PCM 16-bit tiap 1 detik ke /api/sesi/<id>/audio
// -> server menilai jendela 5 detik -> tampilkan status -> putar teguran bila perlu.
// =====================================================================
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  info: null,
  sesi: null,
  ctx: null, stream: null, sumber: null, node: null,
  antre: [], jmlAntre: 0, sedangKirim: false,
  memutar: false, aktif: false,
  riwayat: [],            // {t, prob, tegur}
  mulai: 0, nilaiTotal: 0, nilaiBerisik: 0, teguran: 0,
  bufferSuara: null, wakeLock: null, timerDurasi: null,
};
const SIMPAN = "penjaga-tenang-lab:pengaturan";

// ---------------------------------------------------------------------
// Utilitas
// ---------------------------------------------------------------------
function jam(d = new Date()) {
  return d.toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function mmss(detik) {
  detik = Math.max(0, Math.round(detik));
  const h = Math.floor(detik / 3600), m = Math.floor((detik % 3600) / 60), s = detik % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
}
function pesan(id, teks, galat = false) {
  const el = $(id); el.textContent = teks || ""; el.classList.toggle("galat", !!galat);
}
async function api(url, opsi = {}) {
  const r = await fetch(url, opsi);
  let data = null;
  try { data = await r.json(); } catch (_) { /* bukan JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}
function css(nama) { return getComputedStyle(document.documentElement).getPropertyValue(nama).trim(); }

// ---------------------------------------------------------------------
// Pengaturan
// ---------------------------------------------------------------------
function bacaPengaturan() {
  return {
    ambang: parseFloat($("ambang").value),
    konfirmasi: parseInt($("konfirmasi").value, 10),
    riwayat: parseInt($("riwayat").value, 10),
    cooldown: parseFloat($("cooldown").value),
    volume: parseFloat($("volume").value),
    mic: $("pilihMic").value,
  };
}
function tampilkanNilai() {
  const p = bacaPengaturan();
  $("nilaiAmbang").textContent = p.ambang.toFixed(2);
  $("nilaiCooldown").textContent = p.cooldown + " dtk";
  $("nilaiVolume").textContent = Math.round(p.volume * 100) + "%";
  $("garisAmbang").style.left = (p.ambang * 100) + "%";
  if (p.konfirmasi > p.riwayat) { $("konfirmasi").value = p.riwayat; }
  gambarTitik([]);
}
let timerSimpan = null;
function pengaturanBerubah() {
  tampilkanNilai();
  try { localStorage.setItem(SIMPAN, JSON.stringify(bacaPengaturan())); } catch (_) {}
  clearTimeout(timerSimpan);
  timerSimpan = setTimeout(async () => {
    if (!S.sesi) return;
    try { await api(`/api/sesi/${S.sesi}/pengaturan`, jsonPost(bacaPengaturan())); }
    catch (e) { pesan("pesanMic", "Gagal menyimpan pengaturan: " + e.message, true); }
  }, 250);
  gambarGrafik();
}
function muatPengaturanTersimpan() {
  let p = null;
  try { p = JSON.parse(localStorage.getItem(SIMPAN) || "null"); } catch (_) {}
  if (!p) return false;
  for (const k of ["ambang", "konfirmasi", "riwayat", "cooldown", "volume"]) {
    if (p[k] !== undefined) $(k).value = p[k];
  }
  S.micTersimpan = p.mic || "";
  return true;
}
const jsonPost = (obj) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) });

// ---------------------------------------------------------------------
// Info model
// ---------------------------------------------------------------------
async function muatInfo() {
  try {
    S.info = await api("/api/info");
  } catch (e) {
    pesan("pesanMic", "Model gagal dimuat: " + e.message, true);
    return;
  }
  if (!muatPengaturanTersimpan()) $("ambang").value = S.info.ambang_bawaan;
  tampilkanNilai();
}

async function muatRiwayatServer() {
  try {
    const data = await api("/api/riwayat");
    if (data.length) {
      $("daftarTeguran").innerHTML = data.map((r) =>
        `<li>${new Date(r.waktu).toLocaleString("id-ID")} <span>prob ${r.prob.toFixed(2)}</span></li>`).join("");
    }
  } catch (_) {}
}

// ---------------------------------------------------------------------
// Mikrofon
// ---------------------------------------------------------------------
async function daftarMic() {
  try {
    const perangkat = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
    const sel = $("pilihMic"), pilihan = sel.value || S.micTersimpan || "";
    sel.innerHTML = `<option value="">Default sistem</option>` + perangkat.map((d, k) =>
      `<option value="${d.deviceId}">${d.label || "Mikrofon " + (k + 1)}</option>`).join("");
    if ([...sel.options].some((o) => o.value === pilihan)) sel.value = pilihan;
  } catch (_) {}
}

async function mulai() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    pesan("pesanMic", "Mikrofon hanya bisa dipakai lewat http://localhost atau HTTPS. Buka halaman ini di PC lab dengan alamat localhost.", true);
    return;
  }
  const tbl = $("tblMulai"); tbl.disabled = true;
  try {
    S.ctx = new AudioContext();
    await S.ctx.audioWorklet.addModule("/static/perekam-worklet.js");
    const p = bacaPengaturan();
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: p.mic ? { exact: p.mic } : undefined,
        channelCount: 1,
        // dimatikan agar suara asli lab tidak diubah browser (fitur RMS bergantung pada volume asli)
        echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      },
    });
    S.sumber = S.ctx.createMediaStreamSource(S.stream);
    S.node = new AudioWorkletNode(S.ctx, "perekam");
    const senyap = S.ctx.createGain(); senyap.gain.value = 0;
    S.sumber.connect(S.node); S.node.connect(senyap); senyap.connect(S.ctx.destination);
    S.node.port.onmessage = (e) => terimaSampel(e.data);

    const r = await api("/api/sesi", jsonPost(p));
    S.sesi = r.sesi;
    S.aktif = true; S.antre = []; S.jmlAntre = 0;
    S.mulai = Date.now(); S.nilaiTotal = S.nilaiBerisik = 0;
    S.timerDurasi = setInterval(perbaruiStatistik, 1000);
    await daftarMic();
    await siapkanSuara();
    try { S.wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) {}

    const label = S.stream.getAudioTracks()[0]?.label || "mikrofon";
    pesan("pesanMic", `Mendengarkan: ${label} (${S.ctx.sampleRate} Hz)`);
    setStatus("kumpul", "MENDENGAR", "Mengumpulkan 5 detik audio pertama…");
    tbl.textContent = "Berhenti"; tbl.classList.add("berhenti");
  } catch (e) {
    let teks = e.message;
    if (e.name === "NotAllowedError") teks = "Izin mikrofon ditolak. Izinkan mikrofon di pengaturan situs browser.";
    if (e.name === "NotFoundError" || e.name === "OverconstrainedError") teks = "Mikrofon tidak ditemukan. Pilih mikrofon lain.";
    pesan("pesanMic", teks, true);
    await berhenti(true);
  } finally {
    tbl.disabled = false;
  }
}

async function berhenti(diam = false) {
  S.aktif = false;
  clearInterval(S.timerDurasi);
  try { S.stream?.getTracks().forEach((t) => t.stop()); } catch (_) {}
  try { await S.ctx?.close(); } catch (_) {}
  try { await S.wakeLock?.release(); } catch (_) {}
  S.ctx = S.stream = S.node = S.sumber = S.wakeLock = null;
  S.bufferSuara = null;
  const tbl = $("tblMulai"); tbl.textContent = "Mulai mendengarkan"; tbl.classList.remove("berhenti");
  setStatus("siaga", "SIAGA", "Tekan Mulai untuk mendengarkan lab");
  $("levelIsi").style.width = "0"; $("levelAngka").textContent = "– dBFS";
  if (!diam) pesan("pesanMic", "Berhenti mendengarkan.");
}

let levelPuncak = 0, levelWaktu = 0;
function terimaSampel(blok) {
  if (!S.aktif) return;
  // meter level langsung di browser
  let jumlah = 0; for (let i = 0; i < blok.length; i++) jumlah += blok[i] * blok[i];
  levelPuncak = Math.max(levelPuncak, Math.sqrt(jumlah / blok.length));
  const kini = performance.now();
  if (kini - levelWaktu > 100) {
    const db = 20 * Math.log10(levelPuncak + 1e-10);
    $("levelIsi").style.width = Math.max(0, Math.min(100, (db + 80) / 80 * 100)) + "%";
    $("levelAngka").textContent = db.toFixed(0) + " dBFS";
    levelPuncak = 0; levelWaktu = kini;
  }
  if (S.memutar) return;              // abaikan audio saat teguran diputar
  S.antre.push(blok); S.jmlAntre += blok.length;
  if (S.jmlAntre >= S.ctx.sampleRate && !S.sedangKirim) kirim();
}

async function kirim() {
  const n = S.jmlAntre, pcm = new Int16Array(n);
  let o = 0;
  for (const b of S.antre) for (let i = 0; i < b.length; i++) {
    const v = Math.max(-1, Math.min(1, b[i])); pcm[o++] = v < 0 ? v * 32768 : v * 32767;
  }
  S.antre = []; S.jmlAntre = 0; S.sedangKirim = true;
  try {
    const r = await api(`/api/sesi/${S.sesi}/audio?sr=${S.ctx.sampleRate}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm.buffer,
    });
    if (S.aktif) tanganiHasil(r);
  } catch (e) {
    if (S.aktif) {
      pesan("pesanMic", "Gagal menghubungi server: " + e.message, true);
      if (/Sesi tidak ditemukan/.test(e.message)) { const r = await api("/api/sesi", jsonPost(bacaPengaturan())); S.sesi = r.sesi; }
    }
  } finally {
    S.sedangKirim = false;
  }
}

function tanganiHasil(r) {
  if (!r.siap) {
    setStatus("kumpul", "MENDENGAR", `Mengumpulkan audio… ${r.terkumpul}/${r.dibutuhkan} detik`);
    return;
  }
  S.nilaiTotal++; if (r.status === "BERISIK") S.nilaiBerisik++;
  const berisik = r.status === "BERISIK";
  setStatus(berisik ? "berisik" : "tenang", berisik ? "BERISIK" : "TENANG",
    r.tegur ? "Lab ramai, teguran diputar"
      : berisik ? `${r.riwayat_berisik} dari ${r.riwayat_isi} penilaian terakhir berisik`
      : "Suasana lab kondusif");
  $("angkaProb").textContent = r.prob.toFixed(2);
  $("isiMeter").style.width = (r.prob * 100) + "%";
  $("kartuStatus").style.setProperty("--p", r.prob * 100);      // isi cincin gauge
  S.riwayat.push({ t: Date.now(), prob: r.prob, tegur: r.tegur });
  const batas = Date.now() - 180000;
  while (S.riwayat.length && S.riwayat[0].t < batas) S.riwayat.shift();
  gambarTitik(S.riwayat.slice(-bacaPengaturan().riwayat).map((x) => x.prob >= r.ambang));
  gambarGrafik();
  pesan("pesanMic", `Penilaian ${jam()} · ${r.waktu_ms} ms · level ${r.level_db} dBFS`);
  if (r.tegur) tegur(r.prob);
}

function setStatus(kunci, kata, ket) {
  $("kartuStatus").dataset.status = kunci;
  $("kataStatus").textContent = kata;
  $("ketStatus").textContent = ket;
  if (kunci === "siaga" || kunci === "kumpul") {
    $("angkaProb").textContent = "–"; $("isiMeter").style.width = "0";
    $("kartuStatus").style.setProperty("--p", 0);
  }
}
function gambarTitik(nilai) {
  const n = parseInt($("riwayat").value, 10), el = $("titikRiwayat");
  el.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const t = document.createElement("i");
    const v = nilai[nilai.length - n + i];
    if (v === true) t.className = "ya"; else if (v === false) t.className = "tidak";
    el.appendChild(t);
  }
}
function perbaruiStatistik() {
  $("stDurasi").textContent = mmss((Date.now() - S.mulai) / 1000);
  $("stPersen").textContent = S.nilaiTotal ? Math.round(100 * S.nilaiBerisik / S.nilaiTotal) + "%" : "0%";
  $("stTeguran").textContent = S.teguran;
}

// ---------------------------------------------------------------------
// Suara teguran
// ---------------------------------------------------------------------
async function siapkanSuara() {
  if (!S.ctx || !S.info?.ada_suara || S.bufferSuara) return;
  try {
    const data = await (await fetch("/suara/peringatan.wav", { cache: "no-store" })).arrayBuffer();
    S.bufferSuara = await S.ctx.decodeAudioData(data);
  } catch (_) { S.bufferSuara = null; }
}

function putarBuffer(ctx, buffer, volume) {
  return new Promise((selesai) => {
    const src = ctx.createBufferSource(), g = ctx.createGain();
    src.buffer = buffer; g.gain.value = volume;
    src.connect(g); g.connect(ctx.destination);
    src.onended = selesai; src.start();
  });
}

function desis(ctx, volume, durasi = 1.3) {
  // "sssttt" sintetis: noise putih disaring di pita 4-8 kHz
  const n = Math.floor(ctx.sampleRate * durasi), buf = ctx.createBuffer(1, n, ctx.sampleRate), d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
  const src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
  src.buffer = buf; f.type = "bandpass"; f.frequency.value = 5800; f.Q.value = 0.9;
  const t = ctx.currentTime;
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(0.5 * volume, t + 0.08);
  g.gain.setValueAtTime(0.5 * volume, t + durasi - 0.3);
  g.gain.linearRampToValueAtTime(0, t + durasi);
  src.connect(f); f.connect(g); g.connect(ctx.destination);
  return new Promise((selesai) => { src.onended = selesai; src.start(); });
}

function ucapkan(teks, volume) {
  return new Promise((selesai) => {
    if (!("speechSynthesis" in window)) return selesai();
    const u = new SpeechSynthesisUtterance(teks);
    u.lang = "id-ID"; u.rate = 0.9; u.volume = Math.min(1, volume);
    const suara = speechSynthesis.getVoices().find((v) => v.lang && v.lang.toLowerCase().startsWith("id"));
    if (suara) u.voice = suara;
    const batas = setTimeout(selesai, 5000);
    u.onend = u.onerror = () => { clearTimeout(batas); selesai(); };
    speechSynthesis.speak(u);
  });
}

async function putarTeguran() {
  const vol = bacaPengaturan().volume;
  let ctx = S.ctx, sementara = false;
  if (!ctx) { ctx = new AudioContext(); sementara = true; }
  if (ctx.state === "suspended") await ctx.resume();
  if (!S.bufferSuara && S.info?.ada_suara) {
    try {
      const data = await (await fetch("/suara/peringatan.wav", { cache: "no-store" })).arrayBuffer();
      const b = await ctx.decodeAudioData(data);
      if (!sementara) S.bufferSuara = b;
      await putarBuffer(ctx, b, vol);
      if (sementara) ctx.close();
      return;
    } catch (_) { /* lanjut ke suara bawaan */ }
  }
  if (S.bufferSuara) await putarBuffer(ctx, S.bufferSuara, vol);
  else { await desis(ctx, vol); await ucapkan("jangan ramai", vol); }
  if (sementara) ctx.close();
}

async function tegur(prob) {
  S.memutar = true;
  S.teguran++; perbaruiStatistik();
  $("tirai").hidden = false;
  const li = document.createElement("li");
  li.innerHTML = `${new Date().toLocaleString("id-ID")} <span>prob ${prob.toFixed(2)}</span>`;
  const daftar = $("daftarTeguran");
  daftar.querySelector(".kosong")?.remove();
  daftar.prepend(li);
  try { await putarTeguran(); } catch (_) {}
  await new Promise((r) => setTimeout(r, 600));
  $("tirai").hidden = true;
  // buang audio yang terekam selama teguran agar sistem tidak mendeteksi suaranya sendiri
  S.antre = []; S.jmlAntre = 0;
  try { if (S.sesi) await api(`/api/sesi/${S.sesi}/reset`, { method: "POST" }); } catch (_) {}
  S.memutar = false;
  if (S.aktif) setStatus("kumpul", "MENDENGAR", "Mengumpulkan ulang 5 detik audio…");
}

// ---------------------------------------------------------------------
// Grafik (canvas, tanpa library agar bisa jalan offline di lab)
// ---------------------------------------------------------------------
function siapkanCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight || 180;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return { g, w, h };
}

function gambarSumbu(g, w, h, kiri, bawah, ambang) {
  const warnaGaris = css("--garis"), warnaRedup = css("--redup");
  g.font = "11px system-ui, sans-serif"; g.fillStyle = warnaRedup; g.strokeStyle = warnaGaris; g.lineWidth = 1;
  for (const v of [0, 0.5, 1]) {
    const y = 8 + (1 - v) * (h - 8 - bawah);
    g.beginPath(); g.moveTo(kiri, y); g.lineTo(w, y); g.stroke();
    g.fillText(v.toFixed(1), 2, y + 4);
  }
  const ya = 8 + (1 - ambang) * (h - 8 - bawah);
  g.setLineDash([5, 4]); g.strokeStyle = warnaRedup; g.beginPath(); g.moveTo(kiri, ya); g.lineTo(w, ya); g.stroke(); g.setLineDash([]);
}

function gambarSeri(g, titik, kiri, w, h, bawah, xDari, xKe) {
  const X = (t) => kiri + (t - xDari) / (xKe - xDari) * (w - kiri);
  const Y = (p) => 8 + (1 - p) * (h - 8 - bawah);
  if (!titik.length) return;
  const aksen = css("--aksen");
  g.beginPath(); titik.forEach((d, i) => (i ? g.lineTo(X(d.t), Y(d.prob)) : g.moveTo(X(d.t), Y(d.prob))));
  g.lineTo(X(titik[titik.length - 1].t), Y(0)); g.lineTo(X(titik[0].t), Y(0)); g.closePath();
  g.globalAlpha = 0.12; g.fillStyle = aksen; g.fill(); g.globalAlpha = 1;
  g.beginPath(); titik.forEach((d, i) => (i ? g.lineTo(X(d.t), Y(d.prob)) : g.moveTo(X(d.t), Y(d.prob))));
  g.strokeStyle = aksen; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();
  g.fillStyle = css("--berisik");
  titik.filter((d) => d.tegur).forEach((d) => { g.beginPath(); g.arc(X(d.t), Y(d.prob), 5, 0, Math.PI * 2); g.fill(); });
}

function gambarGrafik() {
  const cv = $("grafik"); if (!cv.clientWidth) return;
  const { g, w, h } = siapkanCanvas(cv), kiri = 28, bawah = 18, kini = Date.now();
  gambarSumbu(g, w, h, kiri, bawah, bacaPengaturan().ambang);
  g.fillStyle = css("--redup");
  for (const m of [3, 2, 1, 0]) {
    const x = kiri + (1 - m / 3) * (w - kiri);
    g.textAlign = m === 0 ? "right" : "center"; g.fillText(m ? `-${m} mnt` : "sekarang", x, h - 3);
  }
  g.textAlign = "left";
  gambarSeri(g, S.riwayat, kiri, w, h, bawah, kini - 180000, kini);
  if (!S.riwayat.length) { g.fillText("Grafik muncul setelah mulai mendengarkan.", kiri + 10, h * 0.3); }
}

// ---------------------------------------------------------------------
// Inisialisasi
// ---------------------------------------------------------------------
function pasang() {
  $("tblMulai").addEventListener("click", () => (S.aktif ? berhenti() : mulai()));
  $("tblTes").addEventListener("click", async () => {
    $("tblTes").disabled = true; $("tirai").hidden = false;
    try { await putarTeguran(); } finally { $("tirai").hidden = true; $("tblTes").disabled = false; }
  });
  $("tirai").addEventListener("click", () => { $("tirai").hidden = true; });
  for (const id of ["ambang", "konfirmasi", "riwayat", "cooldown", "volume"]) $(id).addEventListener("input", pengaturanBerubah);
  $("pilihMic").addEventListener("change", async () => {
    pengaturanBerubah();
    if (S.aktif) { await berhenti(true); await mulai(); }
  });
  window.addEventListener("resize", gambarGrafik);
  if ("speechSynthesis" in window) speechSynthesis.getVoices();   // memicu pemuatan daftar suara
  navigator.mediaDevices?.addEventListener?.("devicechange", daftarMic);
}

pasang();
tampilkanNilai();
muatInfo().then(() => { gambarGrafik(); daftarMic(); });
muatRiwayatServer();