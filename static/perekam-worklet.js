// AudioWorklet: meneruskan sampel mentah mikrofon (mono) ke halaman utama.
class Perekam extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("perekam", Perekam);
