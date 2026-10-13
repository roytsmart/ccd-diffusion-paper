// The track browser and the frame gallery of the documentation.
//
// Both pages are static: the build writes every track, its cutout, and its
// fit to _static/browser/tracks.json and one rendered frame per camera and
// campaign to _static/browser/frames/, and this script draws them. Nothing
// here is computed in Python; the model fractions shown beside each cutout
// are re-evaluated from the fit parameters in the same way as
// ccd_diffusion.tracks.fractions, so that a discrepancy would be visible.

(function () {
  const base = document.currentScript
    ? document.currentScript.src.replace(/[^/]*$/, "")
    : "_static/";

  // ---- the model ----------------------------------------------------------

  // Abramowitz and Stegun 7.1.26, good to 1.5e-7, plenty for a pixel fraction
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const poly =
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592) *
      t;
    return sign * (1 - poly * Math.exp(-x * x));
  }

  // the width of the charge cloud in pixels at fractional depth t
  function widthPixels(t, tc, sm, sd, pixel) {
    const wedge = t < tc ? Math.max(1 - t / Math.max(tc, 1e-6), 0) : 0;
    const g = Math.min((1 - t) / Math.max(1 - tc, 1e-6), 1);
    return Math.sqrt(sm * sm * wedge + sd * sd * g) / pixel;
  }

  // the fraction of a slice's charge in each of the 2h+1 pixels
  function fractions(x, w, slope, h) {
    const s = Math.sqrt(Math.max(w, 1e-3) ** 2 + (slope * slope) / 12);
    const f = [];
    let sum = 0;
    let previous = erf((-h - 0.5 - x) / (s * Math.SQRT2)) / 2;
    for (let j = 0; j < 2 * h + 1; j++) {
      const edge = -h + 0.5 + j;
      const cdf = erf((edge - x) / (s * Math.SQRT2)) / 2;
      f.push(cdf - previous);
      sum += cdf - previous;
      previous = cdf;
    }
    return f.map((v) => v / Math.max(sum, 1e-9));
  }

  // everything the detail view needs about one track, slice by slice
  function analyse(track, meta) {
    const n = track.length;
    const h = meta.half_width;
    const slices = [];
    for (let i = 0; i < n; i++) {
      const charge = track.charge[i];
      const signal = charge.reduce((a, b) => a + b, 0);
      let t = (i + 0.5) / n;
      if (track.orientation < 0) t = 1 - t;
      const x = track.position[i] + track.offset + track.tilt * (i - n / 2);
      const w = widthPixels(t, track.tc, track.sm, track.sd, meta.width_pixel);
      const model = fractions(x, w, track.slope, h);
      const measured = charge.map((q) => q / Math.max(signal, 1e-9));
      const error = track.noise / Math.max(signal, 1e-9);
      const p = measured.reduce((a, f) => a + f * f, 0) - (2 * h + 1) * error * error;
      const pModel = model.reduce((a, f) => a + f * f, 0);
      slices.push({
        depth: t,
        signal: signal,
        usable: signal >= meta.charge_minimum,
        position: x,
        width: w,
        model: model,
        measured: measured,
        p: p,
        pModel: pModel,
      });
    }
    return slices;
  }

  // ---- drawing --------------------------------------------------------------

  // a cutout as an image, slices along x with the back surface at the left,
  // dark for charge, clipped at the 98th percentile like the article's gallery
  function drawCutout(canvas, track, scale, values, overlay) {
    const n = track.length;
    const width = values[0].length;
    canvas.width = n * scale;
    canvas.height = width * scale;
    const ctx = canvas.getContext("2d");
    const flat = values.flat().slice().sort((a, b) => a - b);
    const top = Math.max(flat[Math.floor(0.98 * (flat.length - 1))], 1);
    for (let i = 0; i < n; i++) {
      const slice = track.orientation < 0 ? n - 1 - i : i;
      for (let j = 0; j < width; j++) {
        const v = Math.min(Math.max(values[slice][j] / top, 0), 1);
        const shade = Math.round(255 * (1 - v));
        ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
        ctx.fillRect(i * scale, j * scale, scale, scale);
      }
    }
    if (overlay) {
      ctx.strokeStyle = "rgba(214,39,40,0.9)";
      ctx.lineWidth = Math.max(1, scale / 6);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const slice = track.orientation < 0 ? n - 1 - i : i;
        const y = (overlay[slice] + (width - 1) / 2 + 0.5) * scale;
        const x = (i + 0.5) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  // a small line or bar chart of one or more series against the slice depth
  function drawChart(canvas, series, options) {
    const w = (canvas.width = options.width || 420);
    const h = (canvas.height = options.height || 150);
    const ctx = canvas.getContext("2d");
    const left = 44;
    const right = 8;
    const top = 8;
    const bottom = 26;
    const ys = series.flatMap((s) => s.values).filter((v) => isFinite(v));
    let lo = options.min !== undefined ? options.min : Math.min(...ys);
    let hi = options.max !== undefined ? options.max : Math.max(...ys);
    if (hi <= lo) hi = lo + 1;
    const px = (t) => left + t * (w - left - right);
    const py = (v) => top + (1 - (v - lo) / (hi - lo)) * (h - top - bottom);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#999";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, w - left - right, h - top - bottom);
    ctx.fillStyle = "#333";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(hi.toFixed(options.digits || 0), left - 4, top + 10);
    ctx.fillText(lo.toFixed(options.digits || 0), left - 4, h - bottom);
    ctx.textAlign = "center";
    ctx.fillText("t = z / D", (left + w - right) / 2, h - 6);
    ctx.fillText("0", left, h - 12);
    ctx.fillText("1", w - right, h - 12);
    if (options.reference !== undefined) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#999";
      ctx.beginPath();
      ctx.moveTo(px(0), py(options.reference));
      ctx.lineTo(px(1), py(options.reference));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      if (s.bars) {
        const bw = (w - left - right) / s.values.length;
        s.values.forEach((v, i) => {
          const x = px(s.depths[i]) - bw / 2;
          ctx.globalAlpha = s.faded && s.faded[i] ? 0.3 : 0.8;
          ctx.fillRect(x, py(v), bw - 1, py(lo) - py(v));
        });
        ctx.globalAlpha = 1;
      } else {
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        s.values.forEach((v, i) => {
          if (!isFinite(v)) return;
          const x = px(s.depths[i]);
          const y = py(Math.min(Math.max(v, lo), hi));
          if (!started) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          started = true;
        });
        ctx.stroke();
        if (s.dots) {
          s.values.forEach((v, i) => {
            if (!isFinite(v)) return;
            ctx.beginPath();
            ctx.arc(px(s.depths[i]), py(Math.min(Math.max(v, lo), hi)), 2.5, 0, 2 * Math.PI);
            ctx.fill();
          });
        }
      }
    }
    // legend
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    let lx = left + 6;
    for (const s of series) {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, top + 4, 10, 10);
      ctx.fillStyle = "#333";
      ctx.fillText(s.label, lx + 14, top + 13);
      lx += 14 + ctx.measureText(s.label).width + 14;
    }
  }

  function element(tag, attributes, children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attributes || {})) {
      if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v);
    }
    for (const c of children || []) el.appendChild(c);
    return el;
  }

  // ---- the track browser ------------------------------------------------------

  function browser(root) {
    const status = element("div", { class: "tb-status", text: "Loading the tracks…" });
    root.appendChild(status);
    Promise.all([
      fetch(base + "browser/tracks.json").then((r) => r.json()),
      fetch(base + "browser/frames/frames.json")
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
    ]).then(([data, frames]) => {
      status.remove();
      build(root, data, frames);
    });
  }

  function build(root, data, frames) {
    const keys = Object.keys(data.tracks);
    const tracks = [];
    for (let i = 0; i < data.num; i++) {
      const t = {};
      for (const k of keys) t[k] = data.tracks[k][i];
      tracks.push(t);
    }
    const byName = new Map(tracks.map((t) => [t.name, t]));
    const frameByKey = new Map(frames.map((f) => [f.dataset + "/" + f.fsn, f]));

    const controls = element("div", { class: "tb-controls" });
    const select = (label, id, options) => {
      const s = element("select", { id: id });
      for (const [value, text] of options) s.appendChild(element("option", { value: value, text: text }));
      controls.appendChild(element("label", { text: label + " " }, [s]));
      return s;
    };
    const number = (label, id, value, step) => {
      const n = element("input", { id: id, type: "number", value: value, step: step });
      controls.appendChild(element("label", { text: label + " " }, [n]));
      return n;
    };
    const campaign = select("campaign", "tb-campaign", [["", "all"], ...data.campaigns.map((c) => [c, c])]);
    const chip = select("chip", "tb-chip", [["", "all"], ["FUV1", "FUV1"], ["FUV2", "FUV2"], ["NUV", "NUV"], ["SJI", "SJI"]]);
    const selection = select("selection", "tb-selection", [
      ["all", "every track"],
      ["flat", "flat"],
      ["core", "core (flat, 0.25 < t_c < 0.6)"],
      ["notflat", "not flat"],
      ["loose", "not tight (t_c unconstrained or no diffusion seen)"],
      ["stopping", "stopping (Bragg ratio ≥ 1.5)"],
      ["wide", "wide end to end (t_c > 0.7)"],
    ]);
    const minLength = number("length ≥", "tb-length", 12, 1);
    const tcMin = number("t_c ≥", "tb-tcmin", 0, 0.05);
    const tcMax = number("t_c ≤", "tb-tcmax", 1, 0.05);
    const sort = select("sort by", "tb-sort", [
      ["name", "name"],
      ["length", "length"],
      ["tc", "t_c"],
      ["sm", "σ_max"],
      ["improvement", "improvement over no diffusion"],
      ["bragg", "Bragg ratio"],
      ["sd_preferred", "preferred σ_d"],
      ["fsn", "frame"],
    ]);
    const order = select("order", "tb-order", [["asc", "ascending"], ["desc", "descending"]]);
    const search = element("input", { id: "tb-search", type: "text", placeholder: "name, e.g. 2018may-305" });
    controls.appendChild(element("label", { text: "find " }, [search]));
    root.appendChild(controls);

    const summary = element("div", { class: "tb-summary" });
    root.appendChild(summary);
    const main = element("div", { class: "tb-main" });
    const grid = element("div", { class: "tb-grid" });
    const detail = element("div", { class: "tb-detail" });
    main.appendChild(grid);
    main.appendChild(detail);
    root.appendChild(main);
    const more = element("button", { class: "tb-more", text: "show more" });
    root.appendChild(more);

    const pageSize = 120;
    let shown = 0;
    let current = [];

    function filtered() {
      const c = campaign.value;
      const ch = chip.value;
      const sel = selection.value;
      const len = Number(minLength.value) || 0;
      const lo = Number(tcMin.value);
      const hi = Number(tcMax.value);
      const q = search.value.trim().toLowerCase();
      const out = tracks.filter((t) => {
        if (c && t.dataset !== c) return false;
        if (ch && t.chip !== ch) return false;
        if (t.length < len) return false;
        if (t.tc < lo || t.tc > hi) return false;
        if (q && !t.name.toLowerCase().includes(q)) return false;
        switch (sel) {
          case "flat":
            return t.flat;
          case "core":
            return t.core;
          case "notflat":
            return !t.flat;
          case "loose":
            return !t.tight;
          case "stopping":
            return t.bragg >= 1.5;
          case "wide":
            return !t.crossing;
          default:
            return true;
        }
      });
      const key = sort.value;
      const sign = order.value === "desc" ? -1 : 1;
      out.sort((a, b) => {
        const x = a[key];
        const y = b[key];
        if (typeof x === "string") return sign * x.localeCompare(y, undefined, { numeric: true });
        return sign * (x - y);
      });
      return out;
    }

    function card(t) {
      const canvas = element("canvas", { class: "tb-thumb" });
      drawCutout(canvas, t, 4, t.charge, null);
      const flags = [t.flat ? "flat" : t.tight ? (t.bragg >= 1.5 ? "stopping" : "wide") : "loose"];
      const caption = element("div", {
        class: "tb-caption",
        html:
          `<b>${t.name}</b> ${t.chip}<br>` +
          `t<sub>c</sub> ${t.tc.toFixed(2)}, σ<sub>max</sub> ${t.sm.toFixed(1)} µm, ${t.length} slices, ${flags[0]}`,
      });
      const div = element("div", { class: "tb-card" + (t.flat ? " tb-flat" : "") }, [canvas, caption]);
      div.addEventListener("click", () => show(t));
      return div;
    }

    function render() {
      current = filtered();
      shown = 0;
      grid.innerHTML = "";
      const flat = current.filter((t) => t.flat).length;
      summary.textContent = `${current.length} tracks match, ${flat} of them flat, of ${tracks.length} in the package.`;
      page();
    }

    function page() {
      const next = current.slice(shown, shown + pageSize);
      for (const t of next) grid.appendChild(card(t));
      shown += next.length;
      more.style.display = shown < current.length ? "" : "none";
      more.textContent = `show more (${current.length - shown} left)`;
    }

    function show(t) {
      const slices = analyse(t, data);
      const modelCharge = slices.map((s) => s.model.map((f) => f * s.signal));
      detail.innerHTML = "";
      detail.appendChild(element("h3", { text: t.name }));
      const frame = frameByKey.get(t.dataset + "/" + t.fsn);
      const where = t.vertical ? `column ${t.column}, rows ${t.row} to ${t.row + t.length - 1}` : `row ${t.column}, columns ${t.row} to ${t.row + t.length - 1}`;
      detail.appendChild(
        element("p", {
          class: "tb-meta",
          html:
            `${t.chip}, campaign ${t.dataset}, frame ${t.fsn} at ${t.time}` +
            (frame ? ` (<a href="frames.html#${t.dataset}-${t.fsn}">rendered</a>)` : "") +
            `<br>${where}, ${t.vertical ? "along the columns" : "along the rows"}, slope ${t.slope.toFixed(3)} px per slice`,
        })
      );
      const dataCanvas = element("canvas", { class: "tb-big" });
      drawCutout(dataCanvas, t, 14, t.charge, slices.map((s) => s.position));
      const modelCanvas = element("canvas", { class: "tb-big" });
      drawCutout(modelCanvas, t, 14, modelCharge, null);
      detail.appendChild(element("div", { class: "tb-label", text: "the cutout, back surface at the left, with the fitted centerline" }));
      detail.appendChild(dataCanvas);
      detail.appendChild(element("div", { class: "tb-label", text: "the model, Equation 1 at the fit, with each slice's charge" }));
      detail.appendChild(modelCanvas);

      const rows = [
        ["orientation", t.orientation > 0 ? "enters at the first slice" : "enters at the last slice"],
        ["t_c", `${t.tc.toFixed(3)} [${t.tc_min.toFixed(2)}, ${t.tc_max.toFixed(2)}]`],
        ["σ_max", `${t.sm.toFixed(2)} µm [${t.sm_min.toFixed(1)}, ${t.sm_max.toFixed(1)}]`],
        ["σ_d", `${t.sd.toFixed(2)} µm, pooled over the CCD (this track alone prefers ${t.sd_preferred.toFixed(2)} µm)`],
        ["offset, tilt", `${t.offset.toFixed(3)} px, ${t.tilt.toFixed(4)} px per slice`],
        ["improvement over no diffusion", `${t.improvement.toFixed(1)} units of misfit`],
        ["Bragg ratio", t.bragg.toFixed(2)],
        ["read noise, gain", `${t.noise.toFixed(1)} e⁻, ${t.gain.toFixed(2)} e⁻ per DN`],
        ["selection", `${t.tight ? "tight" : "not tight"}, ${t.bragg < 1.5 ? "no Bragg rise" : "Bragg rise"}, ${t.crossing ? "crossing" : "wide end to end"} → ${t.flat ? (t.core ? "flat, core" : "flat") : "not flat"}`],
      ];
      const table = element("table", { class: "tb-table" });
      for (const [k, v] of rows) table.appendChild(element("tr", {}, [element("th", { text: k }), element("td", { text: v })]));
      detail.appendChild(table);

      const depths = slices.map((s) => s.depth);
      const signal = element("canvas");
      drawChart(
        signal,
        [{ label: "charge per slice (e⁻)", values: slices.map((s) => s.signal), depths: depths, color: "#4c72b0", bars: true, faded: slices.map((s) => !s.usable) }],
        { min: 0, reference: data.charge_minimum, digits: 0 }
      );
      detail.appendChild(element("div", { class: "tb-label", text: "charge per slice; faded slices are below the floor and enter no statistic" }));
      detail.appendChild(signal);
      const p = element("canvas");
      drawChart(
        p,
        [
          { label: "measured Σf²", values: slices.map((s) => (s.usable ? s.p : NaN)), depths: depths, color: "#000", dots: true },
          { label: "model", values: slices.map((s) => s.pModel), depths: depths, color: "#c44e52" },
        ],
        { min: 0, max: 1, digits: 1 }
      );
      detail.appendChild(element("div", { class: "tb-label", text: "the same-pixel probability of each slice against its depth" }));
      detail.appendChild(p);
      const width = element("canvas");
      drawChart(width, [{ label: "fitted width (px)", values: slices.map((s) => s.width), depths: depths, color: "#c44e52" }], { min: 0, digits: 2 });
      detail.appendChild(element("div", { class: "tb-label", text: "the width of the charge cloud along the track" }));
      detail.appendChild(width);
      history.replaceState(null, "", "#" + t.name);
    }

    for (const c of [campaign, chip, selection, minLength, tcMin, tcMax, sort, order]) c.addEventListener("change", render);
    search.addEventListener("input", render);
    more.addEventListener("click", page);
    render();
    const wanted = decodeURIComponent(location.hash.slice(1));
    if (wanted && byName.has(wanted)) show(byName.get(wanted));
    else if (current.length) show(current[0]);
  }

  // ---- the frame gallery -------------------------------------------------------

  function gallery(root) {
    fetch(base + "browser/frames/frames.json")
      .then((r) => r.json())
      .then((frames) => {
        for (const f of frames) {
          const id = `${f.dataset}-${f.fsn}`;
          const camera = f.image === "FUV" ? "spectrograph" : `slit-jaw imager, ${f.image.slice(4)} Å`;
          const section = element("div", { class: "fg-frame", id: id });
          section.appendChild(element("h3", { text: `${f.dataset}: ${camera}, frame ${f.fsn}` }));
          section.appendChild(
            element("p", {
              html:
                `${f.time}, ${f.exposure.toFixed(1)} s exposure, ${f.saa ? "inside" : "outside"} the anomaly, ` +
                `${f.tracks.length} track${f.tracks.length === 1 ? "" : "s"} kept: ` +
                f.tracks.map((n) => `<a href="browser.html#${n}">${n}</a>`).join(", ") +
                `<br>rows ${f.rows[0]} to ${f.rows[1] - 1} and columns ${f.columns[0]} to ${f.columns[1] - 1} of the level-1 image, first row at the bottom; click for full resolution`,
            })
          );
          const src = base + "browser/frames/" + f.file;
          section.appendChild(element("a", { href: src }, [element("img", { src: src, class: "fg-image", loading: "lazy" })]));
          root.appendChild(section);
        }
      });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const b = document.getElementById("track-browser");
    if (b) browser(b);
    const g = document.getElementById("frame-gallery");
    if (g) gallery(g);
  });
})();
