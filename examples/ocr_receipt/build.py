"""Build a synthetic OCR fixture and an annotation viewer; no OCR/model is run."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def build():
    width, height = 560, 660
    image = Image.new("RGB", (width, height), "#fffdf5")
    draw = ImageDraw.Draw(image)
    lines = [
        "NORTHSIDE MARKET",
        "Receipt R-1042",
        "Date: 2026-09-21",
        "Coffee beans       12.00 USD",
        "Oat milk            4.50 USD",
        "Subtotal           16.50 USD",
        "Tax                 1.32 USD",
        "Total              17.82 USD",
        "Payment: CARD",
        "Thank you!",
    ]
    spans = []
    for i, text in enumerate(lines):
        x, y = 35, 35 + i * 57
        box = list(draw.textbbox((x, y), text, font_size=24))
        draw.text((x, y), text, fill="#152438", font_size=24)
        spans.append({"id": f"line-{i}", "text": text, "bbox": box})
    image.save(ROOT / "receipt.png")
    ocr = {
        "provenance": "Synthetic OCR-style transcription from a rendered fixture; not OCR engine output.",
        "image": "receipt.png",
        "width": width,
        "height": height,
        "spans": spans,
    }
    (ROOT / "ocr.json").write_text(json.dumps(ocr, indent=2) + "\n")
    from extract import make_records

    keys = ["merchant", "date", "subtotal", "total", "payment_method", "customer_email"]
    records = make_records(ocr, keys, "receipt.png")
    targets = ["line-0", "line-2", "line-5", "line-7", "line-8", "none"]
    (ROOT / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    labeled = [dict(r, key=k, target=t) for r, k, t in zip(records, keys, targets)]
    (ROOT / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labeled))
    data = json.dumps(labeled).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision-JEV · OCR receipt example</title>
<style>
body{font:16px system-ui;margin:0;background:#eef2f6;color:#152438}main{max-width:1050px;margin:auto;padding:32px}
h1{font-size:32px} .note{padding:16px;background:#fff1cf;border-radius:12px;line-height:1.6}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px}.card{background:white;border-radius:16px;padding:24px}
.visual{position:relative}.visual img{width:100%;display:block}#box{position:absolute;border:3px solid #087e8b;box-sizing:border-box;background:#087e8b22;pointer-events:none}
select{width:100%;padding:12px;font:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px;border-radius:8px}
li{margin:9px 0}a{color:#087e8b}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><main><h1>Vision-JEV / OCR receipt extraction</h1>
<p>Receipt image → OCR spans → candidates → structured extraction</p>
<div class="note"><strong>Annotation explorer — not model inference.</strong> This synthetic fixture uses OCR-style text and boxes, not an OCR engine. Selections below show expected labels. No probabilities, calibration, or speedup are claimed.</div>
<div class="grid"><section class="card"><div class="visual"><img src="receipt.png" alt="Synthetic Northside Market receipt"><div id="box" hidden></div></div></section>
<section class="card"><h2>Request 2–6 keys</h2><div id="keys"></div><p><button id="extract">Extract selected keys</button></p>
<p>Reference preview only. Load a real CLI result to view model probabilities.</p><input type="file" id="load" accept="application/json">
<h2>Structured result</h2><pre id="result"></pre></section></div>
<p><a href="https://github.com/arnodjiang/Vision-JEV">GitHub</a> · <a href="ocr.json">OCR fixture</a> · <a href="requests.jsonl">Inference requests</a> · <a href="annotations.jsonl">Reference annotations</a></p>
</main><script>
const records=DATA;
const keys=document.getElementById('keys');
records.forEach((r,i)=>{const label=document.createElement('label');label.style.display='block';const input=document.createElement('input');input.type='checkbox';input.value=i;input.checked=i<4;label.append(input,document.createTextNode(r.key));keys.append(label)});
function show(){const selected=[...keys.querySelectorAll('input:checked')].map(x=>records[Number(x.value)]);
if(selected.length<2||selected.length>6){document.getElementById('result').textContent='Select 2–6 keys.';return}
const fields={};document.querySelectorAll('.evidence-box').forEach(x=>x.remove());
selected.forEach(r=>{const c=r.candidates.find(c=>c.id===r.target);fields[r.key]={value:c.value??null,confidence:null,confidence_kind:'not_computed',calibrated:false,abstained:c.is_null??false,candidate_id:c.id,evidence:c.bbox?{text:c.text,bbox:c.bbox}:null};
if(c.bbox){const box=document.createElement('div');box.className='evidence-box';const [x1,y1,x2,y2]=c.bbox;Object.assign(box.style,{position:'absolute',border:'2px solid #087e8b',boxSizing:'border-box',left:x1*100+'%',top:y1*100+'%',width:(x2-x1)*100+'%',height:(y2-y1)*100+'%'});box.title=r.key;document.querySelector('.visual').append(box)}});
document.getElementById('result').textContent=JSON.stringify({source:'reference_annotation',fields},null,2)}
document.getElementById('extract').addEventListener('click',show);
document.getElementById('load').addEventListener('change',async e=>{try{const r=JSON.parse(await e.target.files[0].text());if(r.source!=='model'||!r.fields)throw Error('Expected model result JSON');document.querySelectorAll('.evidence-box').forEach(x=>x.remove());document.getElementById('result').textContent=JSON.stringify(r,null,2)}catch(err){document.getElementById('result').textContent=String(err)}});show();
</script>"""
    (ROOT / "index.html").write_text(page.replace("DATA", data))
    print(ROOT / "index.html")


if __name__ == "__main__":
    build()
