/* Browser implementation of the project's fixed structure-v2 feature pipeline. */
(function (root) {
  const EMOTICON = /\[[^\[\]]{1,15}\]/gu;
  const ADDITIVE = /不但|不仅|不光|不只|不止/gu;
  const NON_NEGATION = /特别|分别|区别|告别|别人|别的|别处|别样|别致|别墅|别名|未来|无论|无数/gu;
  const NON_CONTRAST = /但愿|不但|忙不过来/gu;
  const NEGATION = /没有|并非|不是|不能|不会|不得|不要|不|没|未|别|勿|无/gu;
  const CONTRAST = /但是|可是|然而|不过|却|但/gu;
  const BOUNDARY = /[\s，。！？；：,.!?;:、（）()【】\[\]“”"…]+/gu;

  function locations(pattern, text) {
    return Array.from(text.matchAll(pattern), match => ({
      value: match[0],
      start: Array.from(text.slice(0, match.index)).length,
      end: Array.from(text.slice(0, match.index + match[0].length)).length
    }));
  }

  function outside(match, excluded) {
    return !excluded.some(span => match.start < span.end && span.start < match.end);
  }

  function structureV2(text) {
    const clean = text.replace(EMOTICON, ' ');
    const chars = Array.from(clean);
    const negExclusions = locations(ADDITIVE, clean).concat(locations(NON_NEGATION, clean));
    const contrastExclusions = locations(ADDITIVE, clean).concat(locations(NON_CONTRAST, clean));
    const negations = locations(NEGATION, clean).filter(match => outside(match, negExclusions));
    const contrasts = locations(CONTRAST, clean).filter(match => outside(match, contrastExclusions));
    const boundaries = locations(BOUNDARY, clean).map(match => match.start)
      .concat(contrasts.map(match => match.start));
    const stops = boundaries.concat(chars.length).sort((a, b) => a - b);
    const spans = negations.map(match => {
      const stop = stops.find(position => position >= match.end);
      return { marker: match.value, start: match.start,
        following: chars.slice(match.end, Math.min(stop, match.end + 4)).join('') };
    });
    const clauses = new Map();
    for (const span of spans) {
      const clause = boundaries.filter(position => position < span.start).length;
      clauses.set(clause, (clauses.get(clause) || 0) + 1);
    }
    const last = contrasts.at(-1);
    return {
      cleanText: clean,
      spans,
      doubleNegation: Array.from(clauses.values()).some(count => count >= 2),
      before: last ? chars.slice(0, last.start).join('') : '',
      after: last ? chars.slice(last.end).join('') : ''
    };
  }

  function tfidfDot(text, features) {
    // sklearn's char analyzer collapses runs of two or more whitespace characters.
    const chars = Array.from(text.toLowerCase().replace(/\s{2,}/gu, ' '));
    const counts = new Map();
    for (let n = 1; n <= 2; n++) {
      for (let i = 0; i + n <= chars.length; i++) {
        const gram = chars.slice(i, i + n).join('');
        if (Object.prototype.hasOwnProperty.call(features, gram)) {
          counts.set(gram, (counts.get(gram) || 0) + 1);
        }
      }
    }
    let squared = 0;
    let numerator = 0;
    for (const [gram, count] of counts) {
      const [idf, coefficient] = features[gram];
      const value = Math.fround(Math.fround(1 + Math.log(count)) * idf);
      squared += value * value;
      numerator += value * coefficient;
    }
    return squared ? numerator / Math.sqrt(squared) : 0;
  }

  function negationDot(record, features) {
    const values = new Map();
    const add = (key, value = 1) => values.set(key, (values.get(key) || 0) + value);
    add('negation_count_capped_5', Math.min(record.spans.length, 5));
    add('multiple_negations_in_clause', record.doubleNegation ? 1 : 0);
    for (const span of record.spans) {
      add('marker=' + span.marker);
      const chars = Array.from(span.following);
      for (let length = 1; length <= chars.length; length++) {
        add('local=' + span.marker + '|' + chars.slice(0, length).join(''));
      }
    }
    let squared = 0;
    let numerator = 0;
    for (const [key, value] of values) {
      if (!Object.prototype.hasOwnProperty.call(features, key)) continue;
      squared += value * value;
      numerator += value * features[key];
    }
    return squared ? numerator / Math.sqrt(squared) : 0;
  }

  function predict(text, model) {
    if (typeof text !== 'string' || !text.trim() || Array.from(text.trim()).length > 500) {
      throw new Error('请输入 1—500 字的文本。');
    }
    if (!model || model.format !== 'sentiment-structure-v2-js-1') {
      throw new Error('模型格式不正确。');
    }
    const record = structureV2(text.trim());
    const features = model.features;
    const score = model.intercept + tfidfDot(record.cleanText, features.base) +
      negationDot(record, features.negation) + tfidfDot(record.before, features.before) +
      tfidfDot(record.after, features.after);
    return { prediction: score > 0 ? 'positive' : 'negative', score };
  }

  const api = { predict, structureV2 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SentimentInference = api;
})(globalThis);
