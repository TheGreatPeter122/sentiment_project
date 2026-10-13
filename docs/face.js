/*!
 * face.js —— 情感表情可视化组件（笑脸 / 哭脸）
 *
 * 设计约定
 *   输入：情绪极性 prediction（positive / negative）+ 决策分数 score（任意实数）
 *   输出：一张 SVG 表情，嘴角上扬或下垂的角度与 |score| 成正比
 *
 *   嘴角角度 θ = MAX_ANGLE × tanh(|score| / SCORE_REF)        （单位：度）
 *     · 小分数区间近似线性：θ ≈ MAX_ANGLE × |score| / SCORE_REF，即“与数字绝对值成正比”
 *     · 大分数区间用 tanh 平滑封顶，避免个别超长文本把嘴“拉爆”
 *     · SCORE_REF 由本项目真实模型统计得到：|score| 中位数 ≈ 2.6、p75 ≈ 4.5、p90 ≈ 4.9
 *       取 4 时，中等句子约 20°、强烈句子约 35°~44°，视觉区分度最好
 *
 *   曲线：用二次贝塞尔绘制，嘴中点的垂直位移与分数强度严格成正比
 *         curveHeight = sign × DEFLECTION × tanh(|score| / SCORE_REF)   （单位：viewBox 像素）
 *           · 正向 score → sign = −1 → curveHeight < 0 → 嘴中点抬高（嘴角上扬 = 笑脸）
 *           · 负向 score → sign = +1 → curveHeight > 0 → 嘴中点降低（嘴角下垂 = 哭脸）
 *         二次贝塞尔在 t=0.5 处：曲线中点 y = (起点y + 2×控制点y + 终点y) / 4
 *         两端点同高时 ⇒ 曲线中点 y = 基准线 y + curveHeight
 *         ⇒ 控制点 y = 基准线 y + 2 × curveHeight
 *         ⇒ 嘴中点相对两端的位移恒为 |curveHeight| = DEFLECTION × tanh(|score| / SCORE_REF)
 *         两端点固定在 (54, MOUTH_Y) 与 (146, MOUTH_Y)，嘴宽恒为 92
 *
 * 用法
 *   <div id="result"></div>
 *   const face = SentimentFace.create(document.querySelector('#result'));
 *   face.update(score, prediction);          // 静态绘制
 *   face.animateTo(score, prediction, 620);  // 带过渡动画
 *   face.reset();                            // 回到中性表情
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SentimentFace = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  /* ---------------- 可调参数 ---------------- */
  const CONFIG = {
    SCORE_REF: 4,      // 分数参考尺度：|score| = 4 时嘴巴呈现约 76% 的最大幅度
    MAX_ANGLE: 44,     // 嘴角最大偏移角度（度）：0° 平嘴、44° 极限上扬/下垂
    DEFLECTION: 15,    // 嘴中点在 viewBox 内的最大垂直位移（像素）
    SIZE: 208,         // SVG 显示尺寸（viewBox 为 200×200）
    WIDTH: 92,         // 嘴的弦宽（两端点间距）
    MOUTH_Y: 112,      // 嘴两端点的 y 坐标（基准线）
    ANIM_MS: 620       // 默认动画时长
  };

  // 两端点基准线 y → 贝塞尔控制点 y
  //   二次贝塞尔在 t=0.5 处：中点 y = (起点y + 2×控制点y + 终点y) / 4
  //   两端点同高时 ⇒ 中点 y = (2×基准线 y + 2×控制点y) / 4 = (基准线 y + 控制点y) / 2
  //   令中点 y = 基准线 y + curveHeight，解得 控制点 y = 基准线 y + 2 × curveHeight
  function controlPointFor(curveHeight) {
    return CONFIG.MOUTH_Y + 2 * curveHeight;
  }

  // 由控制点 y 反算曲线中点 y（与渲染路径完全一致的口径）
  function mouthMidY(controlY) {
    return (CONFIG.MOUTH_Y + controlY) / 2;
  }

  const THEME = {
    positive: { stroke: '#1f7d4b', fill: '#eef8f0', glow: 'rgba(38,116,72,.20)', word: '偏正向' },
    negative: { stroke: '#b05a34', fill: '#fff4ee', glow: 'rgba(160,82,52,.20)', word: '偏负向' }
  };

  /* ---------------- 纯数学部分（可单独测试） ---------------- */

  // |score| → 数值强度 0~1（tanh 平滑封顶）
  function intensityOf(score, scoreRef) {
    const ref = (Number.isFinite(scoreRef) && scoreRef > 0) ? scoreRef : CONFIG.SCORE_REF;
    const value = Number(score);
    if (!Number.isFinite(value)) return 0;
    const clamped = Math.max(-1e6, Math.min(1e6, value));
    return Math.tanh(Math.abs(clamped) / ref);
  }

  // 强度 → 嘴角角度（度），恒为非负；direction 决定正负方向
  function mouthAngleOf(score, scoreRef) {
    return intensityOf(score, scoreRef) * CONFIG.MAX_ANGLE;
  }

  // 由强度与极性计算嘴的二次贝塞尔几何
  //   curveHeight：嘴中点相对两端基准线的垂直位移
  //     负值 = 中点抬高（嘴角上扬 → 笑脸）；正值 = 中点降低（嘴角下垂 → 哭脸）
  //   其大小严格等于 DEFLECTION × tanh(|score| / SCORE_REF)
  function mouthGeometry(score, prediction, scoreRef) {
    const intensity = intensityOf(score, scoreRef);
    // curveHeight 为“带符号位移”：负值表示嘴中点抬高（正向 / 笑脸），正值表示降低（负向 / 哭脸）
    const sign = prediction === 'negative' ? 1 : -1;
    const half = CONFIG.WIDTH / 2;
    const left = 100 - half;
    const right = 100 + half;
    const base = CONFIG.MOUTH_Y;
    const curveHeight = sign * intensity * CONFIG.DEFLECTION;
    const controlY = controlPointFor(curveHeight);
    return {
      intensity: intensity,
      sign: sign,
      path: `M ${left} ${base} Q 100 ${controlY} ${right} ${base}`,
      controlY: controlY,
      curveHeight: curveHeight,             // 带符号：负=上扬，正=下垂
      peakY: base + curveHeight,            // 嘴中点实际 y（笑脸 < 基准线，哭脸 > 基准线）
      displace: Math.abs(curveHeight),      // 位移量（用于文案展示）
      angle: intensity * CONFIG.MAX_ANGLE
    };
  }

  /* ---------------- SVG 构建 ---------------- */

  function el(name, attrs) {
    const node = document.createElementNS(NS, name);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  }

  function buildSvg() {
    const size = CONFIG.SIZE;
    const svg = el('svg', {
      viewBox: '0 0 200 200',
      width: size, height: size,
      role: 'img',
      'aria-label': '情感表情可视化'
    });
    svg.classList.add('face-svg');

    // 光晕
    const glow = el('circle', { cx: 100, cy: 100, r: 92, fill: 'transparent', class: 'face-glow' });
    // 脸
    const head = el('circle', { cx: 100, cy: 100, r: 74, class: 'face-head' });
    // 眼睛（带高光的圆眼）
    const eyes = el('g', { class: 'face-eyes' });
    for (const cx of [74, 126]) {
      eyes.appendChild(el('circle', { cx: cx, cy: 78, r: 8.2, class: 'face-eye' }));
      eyes.appendChild(el('circle', { cx: cx - 2.6, cy: 75.2, r: 2.5, class: 'face-eye-glint' }));
    }
    // 眉毛
    const brows = el('g', { class: 'face-brows' });
    const browLeft = el('path', { d: 'M 62 56 Q 74 49 87 55', class: 'face-brow' });
    const browRight = el('path', { d: 'M 113 55 Q 126 49 138 56', class: 'face-brow' });
    brows.append(browLeft, browRight);
    // 嘴
    const mouth = el('path', { d: 'M 54 112 Q 100 112 146 112', class: 'face-mouth' });
    mouth.setAttribute('stroke-linecap', 'round');

    svg.append(glow, head, brows, eyes, mouth);
    return { svg, glow, head, browLeft, browRight, mouth };
  }

  /* ---------------- 表情绘制 / 动画 ---------------- */

  function easeInOut(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // 浏览器用 requestAnimationFrame；无 DOM 环境（如 Node 单元测试）退化为定时器
  const G = typeof globalThis !== 'undefined' ? globalThis : {};
  const nextFrame = typeof G.requestAnimationFrame === 'function'
    ? G.requestAnimationFrame.bind(G)
    : (fn) => setTimeout(fn, 16);
  const stopFrame = typeof G.cancelAnimationFrame === 'function'
    ? G.cancelAnimationFrame.bind(G)
    : (id) => clearTimeout(id);

  function Face(rootEl, options) {
    if (!rootEl) throw new Error('SentimentFace: 需要一个容器元素');
    const opt = options || {};
    const parts = buildSvg();

    // 外层容器：表情 + 文字说明
    const wrap = document.createElement('div');
    wrap.className = 'face-wrap';
    const stage = document.createElement('div');
    stage.className = 'face-stage';
    stage.appendChild(parts.svg);
    const meta = document.createElement('div');
    meta.className = 'face-meta';
    const mood = document.createElement('div');
    mood.className = 'face-mood';
    const caption = document.createElement('div');
    caption.className = 'face-caption';
    meta.append(mood, caption);
    wrap.append(stage, meta);
    rootEl.replaceChildren(wrap);

    this.root = rootEl;
    this.el = wrap;
    this.parts = parts;
    this.moodEl = mood;
    this.captionEl = caption;
    this.current = 0;            // 当前动画进度
    this.target = null;          // 目标状态
    this.raf = null;
    this.scoreRef = opt.scoreRef || CONFIG.SCORE_REF;
    this.reset();
    return this;
  }

  // 直接落到目标状态（无动画）
  Face.prototype.update = function (score, prediction) {
    this.target = { score: Number(score), prediction: prediction === 'negative' ? 'negative' : 'positive' };
    this._render(1);
    return this;
  };

  Face.prototype.reset = function () {
    this.target = { score: 0, prediction: 'positive', neutral: true };
    this._render(0);
    this.moodEl.textContent = '等待分析';
    this.captionEl.textContent = '分析后这里会显示情绪表情';
    return this;
  };

  Face.prototype.animateTo = function (score, prediction, duration) {
    const from = this.current;
    this.target = { score: Number(score), prediction: prediction === 'negative' ? 'negative' : 'positive' };
    const ms = Math.max(80, duration || CONFIG.ANIM_MS);
    const start = Date.now();
    if (this.raf !== null) stopFrame(this.raf);
    const step = () => {
      const t = Math.min(1, (Date.now() - start) / ms);
      const eased = easeInOut(t);
      this.current = from + (1 - from) * eased;
      this._render(this.current);
      if (t < 1) {
        this.raf = nextFrame(step);
      } else {
        this.raf = null;
        this._render(1);
      }
    };
    step();
    return this;
  };

  // progress: 0 = 中性平嘴，1 = 完全呈现目标表情
  Face.prototype._render = function (progress) {
    const t = Math.max(0, Math.min(1, progress));
    const target = this.target || { score: 0, prediction: 'positive' };
    const prediction = target.prediction;
    const theme = THEME[prediction];
    const parts = this.parts;
    const half = CONFIG.WIDTH / 2;

    // 目标几何 → 按进度插值控制点（端点固定，避免路径突变）
    const goal = mouthGeometry(target.score, prediction, this.scoreRef);
    const controlY = controlPointFor(0) + (goal.controlY - controlPointFor(0)) * t;
    parts.mouth.setAttribute('d',
      `M ${100 - half} ${CONFIG.MOUTH_Y} Q 100 ${controlY.toFixed(3)} ${100 + half} ${CONFIG.MOUTH_Y}`);

    // 颜色 / 光晕随进度过渡
    parts.head.setAttribute('fill', prediction === 'negative'
      ? `rgba(255, 244, 238, ${t.toFixed(3)})`
      : `rgba(238, 248, 240, ${t.toFixed(3)})`);
    parts.head.setAttribute('stroke', prediction === 'negative'
      ? `rgba(176, 90, 52, ${Math.max(0.25, t).toFixed(3)})`
      : `rgba(31, 125, 75, ${Math.max(0.25, t).toFixed(3)})`);
    parts.mouth.setAttribute('stroke', prediction === 'negative'
      ? `rgba(176, 90, 52, ${Math.max(0.35, t).toFixed(3)})`
      : `rgba(31, 125, 75, ${Math.max(0.35, t).toFixed(3)})`);
    for (const eye of parts.svg.querySelectorAll('.face-eye, .face-brow')) {
      eye.setAttribute('stroke', theme.stroke);
      if (eye.classList.contains('face-eye')) eye.setAttribute('fill', theme.stroke);
    }
    for (const eye of parts.svg.querySelectorAll('.face-eye, .face-eye-glint')) {
      eye.setAttribute('opacity', (0.25 + 0.75 * t).toFixed(3));
    }
    parts.glow.setAttribute('fill', `rgba(${prediction === 'negative' ? '160,82,52' : '38,116,72'}, ${(0.28 * t).toFixed(3)})`);

    this.el.classList.toggle('is-positive', prediction === 'positive');
    this.el.classList.toggle('is-negative', prediction === 'negative');
    this.el.setAttribute('data-prediction', prediction);

    if (!target.neutral) {
      const word = moodWord(goal.angle);
      this.moodEl.textContent = word.text;
      this.moodEl.className = `face-mood ${word.kind}`;
      this.captionEl.innerHTML =
        `<span class="face-score">决策分数 ${Number(target.score).toFixed(3)}</span>` +
        `<span class="face-angle">嘴角角度 ${goal.angle.toFixed(1)}°</span>` +
        `<span class="face-rule">θ = ${CONFIG.MAX_ANGLE}° × tanh(|score| / ${this.scoreRef})` +
        `　｜　嘴中点位移 ${goal.displace.toFixed(1)} / ${CONFIG.DEFLECTION} px</span>`;
    }
  };

  // 强度 → 文字描述（面向课堂讲解）
  function moodWord(angle) {
    const strong = angle >= 34;
    const mild = angle < 14;
    if (strong) return { kind: 'strong', text: '情绪很强：表情幅度接近最大' };
    if (mild) return { kind: 'mild', text: '情绪很弱：几乎是一张平嘴' };
    return { kind: 'medium', text: '情绪中等：嘴角明显变化' };
  }

  Face.prototype.destroy = function () {
    if (this.raf !== null) stopFrame(this.raf);
    this.el.remove();
    this.root.replaceChildren();
  };

  return {
    CONFIG: CONFIG,
    THEME: THEME,
    create: function (rootEl, options) { return new Face(rootEl, options); },
    // 暴露纯函数，便于单元测试与外部复用
    intensityOf: intensityOf,
    mouthAngleOf: mouthAngleOf,
    mouthGeometry: mouthGeometry,
    controlPointFor: controlPointFor,
    mouthMidY: mouthMidY
  };
});
