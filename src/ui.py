"""Look and feel: CSS, animated barbell, count-up stats. Pure presentation helpers."""
from __future__ import annotations
import html

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=DM+Sans:wght@400;500;700&display=swap');
.stApp{background:#EEF1F8;font-family:'DM Sans',system-ui,sans-serif}
.stApp p,.stApp label,.stApp li{font-family:'DM Sans',system-ui,sans-serif}
h1,h2,h3,.disp{font-family:'Bricolage Grotesque','DM Sans',sans-serif!important;letter-spacing:-.02em;color:#121733}
.block-container{max-width:960px;padding-top:1.2rem;padding-bottom:4rem}
#MainMenu,footer{visibility:hidden}header[data-testid=stHeader]{background:transparent}
.logo{font:800 24px 'Bricolage Grotesque';display:flex;gap:10px;align-items:center;color:#121733}
.logo i{width:26px;height:26px;border-radius:8px;background:#3B4BFF;position:relative;display:inline-block}
.logo i:after{content:"";position:absolute;inset:10px 4px;background:#fff;border-radius:2px}
.stButton button,.stDownloadButton button{border-radius:14px;font-weight:700;padding:.6rem 1.2rem;transition:transform .15s,box-shadow .2s,border-color .2s}
.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-2px);box-shadow:0 10px 24px -12px #3B4BFF}
.stButton button:active{transform:scale(.97)}
[class*="st-key-cards"] button{height:auto;min-height:96px;text-align:left;border-radius:18px;padding:16px;align-items:flex-start;white-space:normal;transition:transform .25s cubic-bezier(.3,1.6,.5,1),border-color .2s}
[class*="st-key-cards"] button>div{justify-content:flex-start;text-align:left;width:100%}
[class*="st-key-cards"] button p{margin:0;font-size:14px;line-height:1.35}
[class*="st-key-cards"] button p:first-child{font:700 18px 'Bricolage Grotesque';margin-bottom:4px}
[class*="st-key-cards"] button[kind=primary]{transform:translateY(-3px)}
.hero h1{font-size:clamp(38px,6.5vw,68px);font-weight:800;line-height:1.05;margin:0 0 14px}
.hero p{color:#5B6382;font-size:18px;max-width:46ch}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:34px 0 22px}
.how div{padding:16px 0;border-top:2px solid #121733}.how b{display:block;font:700 18px 'Bricolage Grotesque';margin-bottom:4px}.how span{color:#5B6382}
.chip{display:inline-block;background:#fff;border:1.5px solid #D9DEEE;border-radius:99px;padding:6px 16px;margin:0 8px 8px 0;font-weight:500}
.sum span{display:inline-block;background:#fff;border:1px solid #D9DEEE;border-radius:99px;padding:4px 14px;margin:0 8px 8px 0;font-size:14px}
.step{text-align:center;color:#5B6382;font-size:14px;margin:0 0 16px}
.sub{color:#5B6382;margin:-4px 0 18px}
.bar{width:100%;height:auto;display:block}
.bar .plate{transform-origin:center;transform-box:fill-box}
.bar .plate.anim{animation:drop .7s cubic-bezier(.3,1.4,.5,1) both;animation-delay:var(--d,0s)}
@keyframes drop{from{opacity:0;transform:translateX(var(--dx,30px)) scaleY(.2)}}
.prog{max-width:360px;margin:0 auto}
.load{text-align:center;padding:50px 0}.load .bar{max-width:380px;margin:0 auto 18px}
.warn{background:#FFF1D6;color:#6B4A00;border-radius:16px;padding:14px 18px;margin-bottom:12px;font-size:15px}
.note{background:#fff;border:1px solid #D9DEEE;border-radius:16px;padding:14px 18px;margin-bottom:12px;font-size:15px}
.exn{display:flex;gap:14px;align-items:center}
.badge{flex:none;width:42px;height:42px;border-radius:50%;background:#EEF1F8;display:grid;place-items:center;font:800 18px 'Bricolage Grotesque'}
.exn h3{margin:0;font-size:21px}.meta{color:#5B6382;font-size:14px}
.rx{text-align:right;font:700 20px 'Bricolage Grotesque';white-space:nowrap}.rx small{display:block;font:400 13px 'DM Sans';color:#5B6382}
.tag{display:inline-block;background:#EEF1F8;border-radius:8px;padding:2px 10px;font-size:13px;margin:0 6px 6px 0}
.tag.pr{background:#DCE0FF;font-weight:700}.tag.ml{background:#FFE0D9}
.why{display:grid;grid-template-columns:110px 1fr 38px;gap:8px;align-items:center;font-size:13px;margin:3px 0;color:#5B6382}
.why div.t{height:8px;background:#EEF1F8;border-radius:9px;overflow:hidden}
.why div.t i{display:block;height:100%;border-radius:9px;background:#3B4BFF;transform-origin:left;animation:grow .9s cubic-bezier(.2,.8,.2,1) both}
@keyframes grow{from{transform:scaleX(0)}}
.fun{display:grid;grid-template-columns:200px 1fr 44px;gap:10px;align-items:center;font-size:14px;margin:5px 0}
.fun div.t{height:12px;background:#E3E7F5;border-radius:9px;overflow:hidden}
.fun div.t i{display:block;height:100%;border-radius:9px;background:#FF5A3C;transform-origin:left;animation:grow 1s cubic-bezier(.2,.8,.2,1) both}
@keyframes up{from{opacity:0;transform:translateY(22px)}}
@keyframes flash{0%{box-shadow:0 0 0 0 #FF5A3C99}100%{box-shadow:0 0 0 18px #FF5A3C00}}
[class*="st-key-ex_"]{animation:up .5s cubic-bezier(.2,.8,.2,1) both}
.st-key-flashcard{animation:flash .8s ease-out}
@media(max-width:720px){.how{grid-template-columns:1fr}.fun{grid-template-columns:120px 1fr 38px}.rx{text-align:left}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""


def stagger_css(n: int = 10) -> str:
    return "<style>" + "".join(f".st-key-ex_{i}{{animation-delay:{i * 0.08:.2f}s}}" for i in range(n)) + "</style>"


def h(s: str) -> str:
    """Collapse indentation so Markdown never treats HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def esc(s) -> str:
    return html.escape(str(s))


def barbell(static: int = 0, animate_from: int | None = None, all_plates: int = 5) -> str:
    """Barbell SVG. `static` plates are drawn plain; plates from `animate_from` drop in with a stagger."""
    plates = []
    for i in range(5):
        if i >= all_plates:
            break
        hgt = 54 - i * 9
        col = "#FF5A3C" if i % 2 else "#3B4BFF"
        anim = animate_from is not None and i >= animate_from
        cls = "plate anim" if anim else "plate"
        d = f"{0.15 + (i - (animate_from or 0)) * 0.13:.2f}s" if anim else "0s"
        for side, x, dx in (("l", 120 - i * 12, -(i + 1) * 30), ("r", 231 + i * 12, (i + 1) * 30)):
            plates.append(f'<rect class="{cls}" style="--d:{d};--dx:{dx}px" x="{x}" y="{30 - hgt / 2}" width="9" height="{hgt}" rx="3" fill="{col}"/>')
    return h(f'<svg class="bar" viewBox="0 0 360 60" aria-hidden="true"><rect x="10" y="27" width="340" height="6" rx="3" fill="#D9DEEE"/>'
             f'<rect x="112" y="23" width="6" height="14" rx="2" fill="#5B6382"/><rect x="242" y="23" width="6" height="14" rx="2" fill="#5B6382"/>{"".join(plates)}</svg>')


def stats_component(items: list[tuple[int, str]]) -> str:
    """Self-contained HTML (runs in an iframe) with a count-up animation."""
    colors = ["#3B4BFF", "#FF5A3C", "#121733"]
    tiles = "".join(f'<div style="background:{colors[i % 3]}"><b data-to="{v}">0</b><span>{esc(l)}</span></div>' for i, (v, l) in enumerate(items))
    return f"""<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@800&family=DM+Sans&display=swap" rel="stylesheet">
<style>body{{margin:0;font-family:'DM Sans',sans-serif}}.g{{display:grid;grid-template-columns:repeat({len(items)},1fr);gap:12px}}
.g div{{color:#fff;border-radius:20px;padding:18px;animation:in .6s cubic-bezier(.2,.8,.2,1) both}}
.g b{{font:800 40px 'Bricolage Grotesque';display:block;line-height:1}}.g span{{font-size:14px;opacity:.85}}
@keyframes in{{from{{opacity:0;transform:translateY(16px)}}}}@media(max-width:600px){{.g b{{font-size:28px}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important}}}}</style>
<div class="g">{tiles}</div>
<script>document.querySelectorAll('b').forEach(function(el){{var to=+el.dataset.to,t0=performance.now();
(function f(n){{var k=Math.min(1,(n-t0)/900);el.textContent=Math.round(to*(1-Math.pow(1-k,3)));if(k<1)requestAnimationFrame(f)}})(t0)}});</script>"""
