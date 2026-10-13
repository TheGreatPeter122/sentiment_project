/*!
 * face-photo.js —— 用真实照片做“笑脸 / 哭脸 / 中性”表情可视化（Canvas 2D，无第三方库）
 *
 * 主逻辑（三态直显）
 *   score > 0  → 笑脸图 smile
 *   score = 0  → 中性图 neutral
 *   score < 0  → 哭脸图 cry
 *   过渡：|score| 很小时用 smoothstep 交叉淡入，避免生硬跳变（neutralBand 控制中性带宽）
 *
 * 可选（threeStateImages = false 时）
 *   只用中性图，靠“嘴角剪切”把嘴角压下去合成哭脸，用于没有哭脸素材的情况。
 *
 * 嘴角剪切（warpField）：竖直位移 ∝ (x − 脸心x)，即左右嘴角反向移动，
 *   幅度 = shear × tanh(|score| / SCORE_REF) × 分数符号 × 距离权重。
 *
 * 需要 3 个素材（同目录）：cry.png / neutral.png / smile.png
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SentimentFacePhoto = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /* ================= 可校准参数（用 face-calibrate.html 导出后替换本段） ================= */
  const CONFIG = {
    /* --- 照片里脸盘的几何（归一化：0~1 相对画幅宽 / 高） --- */
    cx: 0.478,         // 脸盘圆心 x / 画幅宽
    cy: 0.500,         // 脸盘圆心 y / 画幅高
    r: 0.470,          // 脸盘半径 / 画幅宽
    /* --- 变形基准 --- */
    line: 0.620,       // 转动中心线（相对脸盘：0 = 脸顶，1 = 脸底）——放在嘴与眼之间
    influence: 0.750,  // 影响范围（相对脸盘半径）：离中心线多远达到最大位移
    shear: 30,         // 嘴角最大竖直位移（像素，基于 512px 画幅，会自动按尺寸缩放）
    facing: 'right',   // 素材中脸朝哪边：'right' | 'left'（决定左右嘴角谁上谁下）
    SCORE_REF: 4,      // 分数参考尺度（|score| = 4 时约 76% 幅度）
    /* --- 三态显示（主逻辑） --- */
    threeStateImages: true,   // true = 直接切换三张照片；false = 只用中性图靠剪切合成哭脸
    neutralBand: 0.2,         // 中性硬范围：|score| ≤ 该值一律显示纯中性图（不混笑脸/哭脸）
    blendGain: 1.2,           // 过渡快慢：混合量 = tanh(|score| × blendGain)，越小中性区越宽、切换越缓
    baseShear: 0,             // 中性图自身的轻微变形（一般 0）
    smileShear: 0,            // 笑脸图是否再剪切（一般 0，照片已经笑了）
    cryShear: 0,              // 哭脸图是否再剪切（一般 0）
    frownShear: 55,           // 没有哭脸图时，用中性图向下剪切的强度
    /* --- 三张图的统一对齐（归一化；先按脸心/半径调好 CONFIG 的 cx,cy,r，再用这里微调） --- */
    align: {
      neutral: { scale: 1.00, dx: 0, dy: 0 },
      smile:   { scale: 1.00, dx: 0, dy: 0 },
      cry:     { scale: 1.00, dx: 0, dy: 0 }
    },
    /* --- 素材与外观 --- */
    assets: { cry: './cry.png', neutral: './neutral.png', smile: './smile.png' },
    faceSize: 208,     // 组件显示尺寸（CSS 像素）
    ANIM_MS: 620
  };

  /* ================= 数学部分（可在 Node 中单独测试） ================= */
  // 强度：|score| 映射到 0~1（tanh 平滑封顶），恒为非负；正负方向由调用处决定
  function intensityOf(score, scoreRef) {
    const ref = (Number.isFinite(scoreRef) && scoreRef > 0) ? scoreRef : CONFIG.SCORE_REF;
    const v = Number(score);
    if (!Number.isFinite(v)) return 0;
    return Math.abs(Math.tanh(Math.max(-1e6, Math.min(1e6, v)) / ref));
  }

  // 变形场：嘴角剪切（"嘴角上扬 / 下垂"的几何本质）
  //   竖直位移 shift = A × w(y) × (x − 圆心x) / 半径
  //     · x 在圆心右边 → 位移方向与左边相反 ⇒ 左右嘴角一上一下 ⇒ 出现笑容 / 哭脸
  //     · 幅度沿水平距离线性增长 ⇒ 嘴角（离圆心最远）动得最多，鼻尖、眼角几乎不动
  //     · w(y) 由到"转动中心线"的距离决定：中心线处为 0，越远越大（受 influence 控制）
  //   A = shear(像素) × tanh(|score| / SCORE_REF)；score>0 与 score<0 方向相反。
  //   facing = 'left' 时整体反向（用于脸朝左的素材）。
  //   返回 Int16Array：偶数下标 = x 位移，奇数下标 = y 位移。
  function warpField(width, height, score, cfg) {
    const c = cfg || CONFIG;
    const R = c.r * height;
    const cx = c.cx * width;
    const lineY = (c.cy - c.r + 2 * c.r * c.line) * height;
    const range = Math.max(8, c.influence * R);
    // 强度取绝对值后必须再乘回分数的正负号：正分 → 嘴角上扬，负分 → 嘴角下垂
    const signed = intensityOf(score, c.SCORE_REF) * Math.sign(Number(score) || 0);
    const amp = c.shear * (height / 512) * signed * (c.facing === 'left' ? -1 : 1);
    const field = new Int16Array(width * height * 2);
    for (let y = 0; y < height; y++) {
      const w = Math.max(0, Math.min(1, Math.abs(y - lineY) / range));
      const rowBase = y * width * 2;
      for (let x = 0; x < width; x++) {
        const shift = Math.round(amp * w * (x - cx) / R);
        field[rowBase + x * 2 + 1] = shift;      // 只动 y
      }
    }
    return field;
  }

  // 兼容旧接口：按行给出竖直位移（单测/调试用）
  function displacementProfile(height, score, cfg) {
    const f = warpField(height, height, score, cfg || CONFIG);
    const out = new Float32Array(height);
    for (let y = 0; y < height; y++) out[y] = f[(y * height + Math.floor(height / 2)) * 2 + 1];
    return out;
  }

  /* ================= 图像处理 ================= */
  const cache = { key: '', data: null };

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      // file:// 下给图片设 crossOrigin 会被 CORS 拦掉，直接不设（代价是画布变为只读，不能 getImageData）
      if (typeof location === 'undefined' || location.protocol !== 'file:') img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('图片加载失败：' + src));
      img.src = src;
    });
  }

  function toCanvas(img, w, h) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    cv.getContext('2d').drawImage(img, 0, 0, w, h);
    return cv;
  }

  // 按变形场重采样：每个目标像素从 (x+dx, y+dy) 取源像素，结果写回一张 canvas
  function displaceToCanvas(sourceData, width, height, field) {
    const src = sourceData.data;
    const out = new Uint8ClampedArray(src.length);
    for (let y = 0; y < height; y++) {
      const rowBase = y * width;
      for (let x = 0; x < width; x++) {
        const fi = (rowBase + x) * 2;
        let sx = x + field[fi];
        let sy = y + field[fi + 1];
        if (sx < 0) sx = 0; else if (sx >= width) sx = width - 1;
        if (sy < 0) sy = 0; else if (sy >= height) sy = height - 1;
        const so = (sy * width + sx) * 4, dofs = (rowBase + x) * 4;
        out[dofs] = src[so];
        out[dofs + 1] = src[so + 1];
        out[dofs + 2] = src[so + 2];
        out[dofs + 3] = src[so + 3];
      }
    }
    const cv = document.createElement('canvas');
    cv.width = width; cv.height = height;
    cv.getContext('2d').putImageData(new ImageData(out, width, height), 0, 0);
    return cv;
  }

  // 取得“变形后的画布”，相同参数只算一次
  //   注意：缓存键必须用“有符号”的分数，否则 +6 与 −6 会撞进同一个条目
  function frameFor(img, height, score, cfg) {
    const c = cfg || CONFIG;
    const key = img.src + '|' + height + '|' + Number(score).toFixed(4) +
      '|' + c.shear + '|' + c.line + '|' + c.influence + '|' + c.facing;
    if (cache.key === key) return cache.data;
    const width = Math.round(height * img.naturalWidth / img.naturalHeight);
    const cv = toCanvas(img, width, height);
    let moved = cv;
    try {
      const src = cv.getContext('2d').getImageData(0, 0, width, height);
      moved = displaceToCanvas(src, width, height, warpField(width, height, score, c));
    } catch (err) {
      // 画布被 CORS 污染（file:// 直接打开时会发生）：跳过变形，直接用原图
      moved = cv;
    }
    cache.key = key;
    cache.data = { source: cv, moved: moved, width: width, height: height };
    return cache.data;
  }

  /* ================= 渲染：三态直显 =================
   *   score > +band  → 笑脸图 smile
   *   |score| ≤ band → 中性图 neutral
   *   score < −band  → 哭脸图 cry
   *   过渡用 smoothstep 交叉淡入淡出（band 控制"多接近 0 才算中性"）
   *   若按脸心对齐后仍有偏差，用 align 里的 scale / dx / dy 微调各张图。
   */
  function drawFrame(g, targetSize, img, height, score, align, c) {
    const f = frameFor(img, height, score, c);
    const a = align || {};
    const s = (a.scale == null ? 1 : a.scale) * targetSize;
    const x = (targetSize - s) / 2 + (a.dx || 0) * targetSize;
    const y = (targetSize - s) / 2 + (a.dy || 0) * targetSize;
    g.drawImage(f.moved, x, y, s, s);
  }

  function renderTo(target, assets, score, size, cfg) {
    const c = cfg || CONFIG;
    const s = Number(score) || 0;
    const targetSize = size || c.faceSize;
    const height = 512;
    target.width = targetSize; target.height = targetSize;
    const g = target.getContext('2d');
    g.clearRect(0, 0, targetSize, targetSize);
    g.globalAlpha = 1;

    const band = Math.max(1e-6, c.neutralBand == null ? 0.02 : c.neutralBand);
    const t = s > band ? 1 : (s < -band ? -1 : 0);
    // 过渡：|score| × blendGain 越大越快切到笑脸/哭脸（score = 0 时混合量为 0 → 纯中性）
    const gain = c.blendGain == null ? 2.5 : c.blendGain;
    const blend = t === 0 ? 0 : Math.tanh(Math.abs(s) * gain);
    const usePhotos = c.threeStateImages !== false;

    const align = c.align || {};
    const baseScore = c.baseShear || 0;                  // 中性图自身的轻微变形

    if (t > 0 && usePhotos && assets.smile) {
      // 笑脸：中性图打底 + 笑脸图淡入
      drawFrame(g, targetSize, assets.neutral, height, baseScore, align.neutral, c);
      g.globalAlpha = blend;
      drawFrame(g, targetSize, assets.smile, height, c.smileShear || 0, align.smile, c);
    } else if (t < 0 && usePhotos && assets.cry) {
      // 哭脸：中性图打底 + 哭脸图淡入
      drawFrame(g, targetSize, assets.neutral, height, baseScore, align.neutral, c);
      g.globalAlpha = blend;
      drawFrame(g, targetSize, assets.cry, height, c.cryShear || 0, align.cry, c);
    } else {
      // 不用照片（或缺少对应素材）：只用中性图，靠向下剪切把嘴角压成哭脸
      let sc = baseScore;
      if (t < 0) sc = -Math.abs(c.frownShear == null ? 55 : c.frownShear) * blend;
      drawFrame(g, targetSize, assets.neutral, height, sc, align.neutral, c);
    }
    g.globalAlpha = 1;
    return { state: t, blend: blend, intensity: Math.abs(t) * blend };
  }

  /* ================= 组件（对外 API 与 face.js 保持一致） ================= */
  const G = typeof globalThis !== 'undefined' ? globalThis : {};
  const nextFrame = typeof G.requestAnimationFrame === 'function' ? G.requestAnimationFrame.bind(G) : (f) => setTimeout(f, 16);
  const stopFrame = typeof G.cancelAnimationFrame === 'function' ? G.cancelAnimationFrame.bind(G) : (i) => clearTimeout(i);

  function Face(rootEl, options) {
    if (!rootEl) throw new Error('SentimentFacePhoto: 需要一个容器元素');
    const opt = options || {};
    this.cfg = Object.assign({}, CONFIG, opt.config || {});
    this.assets = null;
    this.current = 0;
    this.target = { score: 0, prediction: 'positive', neutral: true };
    this.raf = null;

    const wrap = document.createElement('div');
    wrap.className = 'face-wrap face-photo';
    const stage = document.createElement('div');
    stage.className = 'face-stage';
    const canvas = document.createElement('canvas');
    canvas.className = 'face-canvas';
    canvas.width = this.cfg.faceSize; canvas.height = this.cfg.faceSize;
    canvas.style.width = this.cfg.faceSize + 'px';
    canvas.style.height = this.cfg.faceSize + 'px';
    stage.appendChild(canvas);
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
    this.canvas = canvas;
    this.moodEl = mood;
    this.captionEl = caption;
    this.ready = false;

    const a = this.cfg.assets;
    Promise.all([loadImage(a.neutral), loadImage(a.smile), loadImage(a.cry)])
      .then(([neutral, smile, cry]) => {
        this.assets = { neutral: neutral, smile: smile, cry: cry };
        this.ready = true;
        this._paint(1);
      })
      .catch((err) => {
        this.errorEl = document.createElement('div');
        this.errorEl.className = 'face-error';
        this.errorEl.textContent = '表情素材加载失败：' + err.message;
        meta.appendChild(this.errorEl);
      });

    this._paint(0);
    return this;
  }

  Face.prototype.update = function (score, prediction) {
    this.target = { score: Number(score), prediction: prediction === 'negative' ? 'negative' : 'positive' };
    this.current = 1;
    this._paint(1);
    return this;
  };

  Face.prototype.reset = function () {
    this.target = { score: 0, prediction: 'positive', neutral: true };
    this.current = 0;
    this._paint(0);
    this.moodEl.textContent = '等待分析';
    this.moodEl.className = 'face-mood';
    this.captionEl.textContent = '分析后这里会显示情绪表情';
    return this;
  };

  Face.prototype.animateTo = function (score, prediction, duration) {
    const from = this.current;
    this.target = { score: Number(score), prediction: prediction === 'negative' ? 'negative' : 'positive' };
    const ms = Math.max(80, duration || this.cfg.ANIM_MS);
    const start = Date.now();
    if (this.raf !== null) stopFrame(this.raf);
    const step = () => {
      const t = Math.min(1, (Date.now() - start) / ms);
      const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      this.current = from + (1 - from) * eased;
      this._paint(this.current);
      if (t < 1) this.raf = nextFrame(step); else { this.raf = null; this._paint(1); }
    };
    step();
    return this;
  };

  Face.prototype._paint = function (progress) {
    const t = Math.max(0, Math.min(1, progress));
    const info = { state: 0, blend: 0 };
    if (this.ready) {
      const r = renderTo(this.canvas, this.assets, this.target.score * t, this.cfg.faceSize, this.cfg);
      info.state = r.state; info.blend = r.blend;
    }
    if (this.target.neutral) return;
    const word = this._moodWord(this.target.score);
    this.moodEl.textContent = word.text;
    this.moodEl.className = 'face-mood ' + word.kind;
    this.captionEl.innerHTML =
      '<span class="face-score">决策分数 ' + Number(this.target.score).toFixed(3) + '</span>' +
      '<span class="face-angle">' + word.tag + '</span>' +
      '<span class="face-rule">' + word.rule + '</span>';
  };

  // 三态文案
  Face.prototype._moodWord = function (score) {
    const s = Number(score) || 0;
    if (s > 0) return { kind: 'strong', tag: '笑脸 😊', text: '偏正向：显示笑脸', rule: 'score > 0 → 笑脸图' };
    if (s < 0) return { kind: 'negative', tag: '哭脸 😢', text: '偏负向：显示哭脸', rule: 'score < 0 → 哭脸图' };
    return { kind: 'mild', tag: '中性 😐', text: '中性：分数正好为 0', rule: 'score = 0 → 中性图' };
  };

  Face.prototype.destroy = function () {
    if (this.raf !== null) stopFrame(this.raf);
    this.el.remove();
  };

  return {
    CONFIG: CONFIG,
    create: function (rootEl, options) { return new Face(rootEl, options); },
    intensityOf: intensityOf,
    warpField: warpField,
    displacementProfile: displacementProfile,
    renderTo: renderTo,
    loadImage: loadImage
  };
});
