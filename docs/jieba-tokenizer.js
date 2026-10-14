/* jieba 0.42.1 dictionary DAG / accurate mode with HMM=False.
 * Port of jieba's get_DAG, calc and __cut_DAG_NO_HMM. See THIRD_PARTY_NOTICES.md.
 * Training uses the Python package; browser uses its exported, unmodified dictionary.
 */
(function(root) {
  function create(data) {
    const frequencies = data.frequencies;
    const prefixes = new Set();
    for (const word of Object.keys(frequencies)) {
      for (let i = 1; i <= word.length; i++) prefixes.add(word.slice(0, i));
    }
    const logTotal = Math.log(data.total);
    function cutBlock(text) {
      const route = Array(text.length + 1);
      route[text.length] = [0, 0];
      for (let i = text.length - 1; i >= 0; i--) {
        const ends = [];
        for (let j = i; j < text.length; j++) {
          const word = text.slice(i, j + 1);
          if (!prefixes.has(word)) break;
          if (frequencies[word]) ends.push(j);
        }
        if (!ends.length) ends.push(i);
        let best = [-Infinity, -1];
        for (const end of ends) {
          const score = Math.log(frequencies[text.slice(i, end + 1)] || 1) - logTotal + route[end + 1][0];
          if (score > best[0] || (score === best[0] && end > best[1])) best = [score, end];
        }
        route[i] = best;
      }
      const output = [];
      let buffer = '';
      for (let i = 0; i < text.length;) {
        const end = route[i][1] + 1;
        const word = text.slice(i, end);
        if (/^[a-zA-Z0-9]$/.test(word)) buffer += word;
        else {
          if (buffer) { output.push(buffer); buffer = ''; }
          output.push(word);
        }
        i = end;
      }
      if (buffer) output.push(buffer);
      return output;
    }
    return function tokenize(text) {
      const tokens = [];
      for (const block of text.split(/([\u4E00-\u9FD5a-zA-Z0-9+#&._%\-]+)/u)) {
        if (!block) continue;
        const words = /^[\u4E00-\u9FD5a-zA-Z0-9+#&._%\-]+$/u.test(block)
          ? cutBlock(block) : Array.from(block);
        tokens.push(...words.filter(word => /^[\p{L}\p{N}_]+$/u.test(word)));
      }
      return tokens;
    };
  }
  root.JiebaTokenizer = { create };
  if (typeof module !== 'undefined' && module.exports) module.exports = { create };
})(globalThis);
