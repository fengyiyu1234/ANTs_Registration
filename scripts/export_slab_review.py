"""Export source-intensity coronal slab review, without changing source data."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stats.ontology import Ontology
from stats.laminar import cover_ids, _labels_in_sample_path
from stats.plot_slab import _ontology_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((args.run/'config_used.yaml').read_text())
    qc = list(csv.DictReader((args.run/'laminar_slab_qc.csv').open()))
    ont = Ontology.from_json(_ontology_path(cfg, str(args.run)))
    # Broad anatomical landmarks, not an independent segmentation.
    regions = [('Isocortex', 'Isocortex', '#ffe45e'), ('MO', 'MO', '#38cfff'),
               ('SS', 'SS', '#ff66cc'), ('HPF', 'HPF', '#66ef99'),
               ('cc', 'cc', '#ff9933')]
    ids = {name: cover_ids(ont, [acr]) for name, acr, _ in regions}
    colors = {name: tuple(bytes.fromhex(color[1:])) for name, _, color in regions}
    fig, axs = plt.subplots(6, 3, figsize=(15, 23), facecolor='#171717')
    rawfig, rawaxs = plt.subplots(6, 3, figsize=(15, 23), facecolor='#171717')
    manifest = []
    for r, row in enumerate(qc):
        sample = row['sample']; source = Path(cfg['samples'][sample]['dir'])
        ip = list(source.glob('*_fine_20um.nii.gz'))
        if len(ip) != 1:
            raise ValueError(f'{sample}: ambiguous intensity source {ip}')
        lp = Path(_labels_in_sample_path(source))
        ni, nl = nib.load(ip[0]), nib.load(lp)
        if ni.shape != nl.shape or not np.allclose(ni.affine, nl.affine, atol=1e-5):
            raise ValueError(f'{sample}: image/labels grid mismatch')
        img = np.asarray(ni.dataobj, dtype=np.float32)
        lab = np.asarray(nl.dataobj, dtype=np.int32)
        lo, hi = int(row['lo']), int(row['hi'])
        assert 0 <= lo <= hi < img.shape[1]
        z = ni.header.get_zooms()
        if not np.allclose(z, [20,20,20]):
            raise ValueError(f'Unexpected spacing: {z}')
        source_cfgs = list(source.glob('*.yaml'))
        source_cfg = yaml.safe_load(source_cfgs[0].read_text()) if len(source_cfgs)==1 else {}
        # Follow existing plot_slab dorsal-up convention, but decide ONCE for
        # the whole slab; never flip adjacent frames independently.
        slab = lab[:,lo:hi+1,:]
        root = np.isin(slab, ids['Isocortex'])
        brain = slab > 0
        rows = np.arange(img.shape[2])[None,None,:]
        flip = bool((root*rows).sum()/max(root.sum(),1) >
                    (brain*rows).sum()/max(brain.sum(),1))
        support = (brain | (img[:,lo:hi+1,:] > 0)).any(axis=1)
        xx, zz = np.where(support)
        x0,x1=max(0,int(xx.min())-8),min(img.shape[0],int(xx.max())+9)
        z0,z1=max(0,int(zz.min())-8),min(img.shape[2],int(zz.max())+9)
        v=img[x0:x1,lo:hi+1,z0:z1]
        positive=v[v>0]
        bottom,top=np.percentile(positive,[1,99.7])
        dest=args.out/sample;dest.mkdir(exist_ok=True)
        chosen=[lo,(lo+hi)//2,hi]
        for y in range(lo,hi+1):
            plane=img[x0:x1,y,z0:z1].T
            labels=lab[x0:x1,y,z0:z1].T
            if flip: plane,labels=plane[::-1],labels[::-1]
            gray=np.uint8(np.clip((plane-bottom)/max(top-bottom,1e-6),0,1)**0.65*255)
            rgb=np.repeat(gray[:,:,None],3,axis=2)
            rgba=np.zeros((*gray.shape,4),dtype=np.uint8)
            for name,_,color in regions:
                mask=np.isin(labels,ids[name]);edge=mask & ~binary_erosion(mask)
                rgba[edge,:3]=colors[name];rgba[edge,3]=255
            Image.fromarray(gray).save(dest/f'{y:03d}_raw.png')
            Image.fromarray(rgba).save(dest/f'{y:03d}_outline.png')
            combined=rgb.copy();edge=rgba[:,:,3]>0;combined[edge]=rgba[edge,:3]
            if y in chosen:
                c=chosen.index(y)
                for ax, pic in [(axs[r,c],combined),(rawaxs[r,c],gray)]:
                    ax.imshow(pic,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    # Uniform physical scale in every panel; pad rather than stretch.
                    ax.set_xlim(-5,330); ax.set_ylim(340,-5)
                    ax.plot([15,65],[320,320],color='white',lw=2)
                    ax.text(15,310,'1 mm',color='white',fontsize=8)
                    ax.set_title(f"{sample} | {'Control' if row['group']=='a' else 'Experimental'} | "
                                 f"{['Front','Middle','Back'][c]} yr={y}"+
                                 (' | repositioned' if source_cfg.get('sample',{}).get('reposition_plan') else ''),
                                 color='white',fontsize=10)
                    ax.axis('off')
        meta=dict(sample=sample,group=row['group'],lo=lo,hi=hi,chosen=chosen,
                  image=str(ip[0]),labels=str(lp),shape=ni.shape,affine=ni.affine.tolist(),
                  spacing_um=list(map(float,z)),display_vertical_flip=flip,crop_xz=[x0,x1,z0,z1],
                  display_percentiles=[1,99.7],display_limits=[float(bottom),float(top)],gamma=0.65,
                  repositioned=bool(source_cfg.get('sample',{}).get('reposition_plan')))
        manifest.append(meta)
        print(sample, 'planes',lo,hi,'crop',meta['crop_xz'],'flip',flip,flush=True)
        del img,lab,slab,root,brain,v
    for f,name in [(fig,'comparison_outline'),(rawfig,'comparison_raw')]:
        f.suptitle('Coronal sample intensity | front / middle / back of each sampling window\n'
                   'Yellow: isocortex | cyan: MO | pink: SS | green: HPF | orange: corpus callosum\n'
                   '20 um resampled intensity; common scale; window positions are NOT atlas-matched planes',
                   color='white',fontsize=13)
        f.tight_layout(rect=(0,0,1,0.955));f.savefig(args.out/f'{name}.png',dpi=150,facecolor=f.get_facecolor())
        f.savefig(args.out/f'{name}.pdf',facecolor=f.get_facecolor());plt.close(f)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    from PIL import ImageDraw
    for d in manifest:
        sample=d['sample'];mid=d['chosen'][1]
        sheet=Image.new('RGB',(1800,1200),'#161616');draw=ImageDraw.Draw(sheet)
        for i,y in enumerate(range(mid-4,mid+5)):
            raw=Image.open(args.out/sample/f'{y:03d}_raw.png').convert('RGBA')
            edge=Image.open(args.out/sample/f'{y:03d}_outline.png')
            for j,pic in enumerate([raw,Image.alpha_composite(raw,edge)]):
                pic.thumbnail((285,360));x=(i%3)*600+j*300;z=(i//3)*400
                sheet.paste(pic.convert('RGB'),(x,z+28))
                draw.text((x+4,z+7),f'{sample} yr={y} '+('raw' if j==0 else 'outline'),fill='white')
        sheet.save(args.out/f'continuous_{sample}.jpg',quality=92)
    html='''<!doctype html><meta charset="utf-8"><title>皮层切块连续切面复核</title>
<style>body{background:#161616;color:#eee;font:16px system-ui;margin:24px}button,select,input{font:inherit;margin:6px}a{color:#79cfff}.grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px}.card{background:#222;padding:10px}.frame{position:relative;width:100%;height:440px;overflow:auto}.frame img{position:absolute;left:0;top:0;width:auto;height:auto;image-rendering:pixelated}.caption{min-height:50px}p{max-width:1150px;line-height:1.6}</style>
<h1>六只动物：冠状切块连续复核</h1>
<p>黄色：Isocortex；青色：MO；粉色：SS；绿色：HPF；橙色：胼胝体。轮廓来自配准图谱，不能当作组织存在的证据。背景为20 µm重采样的样本强度图，s18为复位后图像。每只动物窗口内使用固定对比度。未标定左右侧；纵向按皮层位置统一显示。</p>
<p>各卡片可独立按20 µm逐层浏览。全局滑块只同步窗口相对位置，不代表同一解剖切面。关闭轮廓检查原图裂缝，再开启检查标签是否跨过空隙。低信号也可能来自染色，不能自动判为缺损。</p>
<a href="comparison_outline.png">18张轮廓总览</a> · <a href="comparison_raw.png">18张纯原图总览</a> · <a href="review_notes.md">复核记录</a>
<details open><summary>已复核的连续切面：每只9层，间隔20 µm</summary><p>
<a href="continuous_s10.jpg">s10：154–162，狭长截面及侧缘凹陷</a> ·
<a href="continuous_s18.jpg">s18：125–133，侧面缺口</a> ·
<a href="continuous_s11.jpg">s11：134–142，斜向暗裂隙</a> ·
<a href="continuous_s8.jpg">s8：163–171，表面V形内陷</a> ·
<a href="continuous_s12q.jpg">s12q：151–159，自表面向内的裂隙</a> ·
<a href="continuous_s12t.jpg">s12t：147–155，切缘及局部强信号</a>。
以上是形态观察，不是自动损伤判定。s11、s12q已有damage配置，本图未叠加该mask，不能据此断言统计遗漏了排除。</p></details>
<div><label><input id="outline" type="checkbox" checked>显示轮廓</label><label>显示放大<select id="zoom"><option>1</option><option selected>2</option><option>3</option></select>倍</label>
<button onclick="sync(0)">前</button><button onclick="sync(.5)">中</button><button onclick="sync(1)">后</button>
窗口位置<input id="global" type="range" min="0" max="100" value="50" oninput="sync(this.value/100)"></div><div class="grid" id="grid"></div>
<script>const data=MANIFEST;const pad=n=>String(n).padStart(3,'0');
function update(i){const d=data[i],n=+document.getElementById('s'+i).value;document.getElementById('r'+i).src=`${d.sample}/${pad(n)}_raw.png`;document.getElementById('o'+i).src=`${d.sample}/${pad(n)}_outline.png`;document.getElementById('n'+i).textContent=`yr=${n}；距窗口前端 ${(n-d.lo)*20} µm / ${(d.hi-d.lo)*20} µm`;}
function step(i,k){const s=document.getElementById('s'+i);s.value=+s.value+k;update(i)}
function sync(t){data.forEach((d,i)=>{document.getElementById('s'+i).value=Math.round(d.lo+t*(d.hi-d.lo));update(i)});}
data.forEach((d,i)=>{document.getElementById('grid').insertAdjacentHTML('beforeend',`<section class="card"><h2>${d.sample} · ${d.group==='a'?'对照':'实验'}${d.repositioned?' · 已复位':''}</h2><div class="caption" id="n${i}"></div><button onclick="step(${i},-1)">−20 µm</button><button onclick="step(${i},1)">+20 µm</button><input id="s${i}" type="range" min="${d.lo}" max="${d.hi}" value="${d.chosen[1]}" oninput="update(${i})"><div class="frame"><img id="r${i}"><img class="outline" id="o${i}"></div></section>`);update(i)});
document.getElementById('outline').onchange=e=>document.querySelectorAll('.outline').forEach(x=>x.style.visibility=e.target.checked?'visible':'hidden');
function zoom(){let z=+document.getElementById('zoom').value;data.forEach((d,i)=>['r','o'].forEach(k=>{document.getElementById(k+i).style.width=(d.crop_xz[1]-d.crop_xz[0])*z+'px'}))}document.getElementById('zoom').onchange=zoom;zoom();
</script>'''
    (args.out/'index.html').write_text(html.replace('MANIFEST',json.dumps(manifest)),encoding='utf-8')


if __name__=='__main__':
    main()
