"""Build editable, responsive mechanism schematics for the project page.

Run from any directory with Python and draw.io Desktop installed. The optional
--export flag renders SVGs and PNG previews with the local draw.io CLI.
"""

from pathlib import Path
import argparse
import base64
import io
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INK = '#263b45'
TEAL = '#237f91'
PALE = '#e9f3f4'
GRAY = '#dce5e8'
ORANGE = '#d97b4e'
GREEN = '#548b73'


class Diagram:
    def __init__(self, name, width, height):
        self.name = name
        self.xml = ET.Element('mxfile', host='drawio')
        page = ET.SubElement(self.xml, 'diagram', name=name)
        model = ET.SubElement(page, 'mxGraphModel', page='1', pageWidth=str(width),
                              pageHeight=str(height), background='#ffffff')
        self.root = ET.SubElement(model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.n = 2
        self.box(0, 0, width, height, fill='#ffffff', stroke='none')

    def cell(self, x, y, w, h, value='', style='', parent='1'):
        cid = str(self.n)
        self.n += 1
        c = ET.SubElement(self.root, 'mxCell', id=cid, value=value,
                          style='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=22;'
                          f'fontColor={INK};' + style, vertex='1', parent=parent)
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w),
                      height=str(h), **{'as': 'geometry'})
        return cid

    def group(self, x, y, w, h):
        return self.cell(x, y, w, h, style='group;pointerEvents=0;')

    def box(self, x, y, w, h, text='', fill=PALE, stroke='#b8ced3', parent='1', extra=''):
        return self.cell(x, y, w, h, text,
                         f'rounded=1;arcSize=10;fillColor={fill};strokeColor={stroke};'
                         'strokeWidth=1.4;spacing=7;' + extra, parent)

    def text(self, x, y, w, h, text, size=22, color=INK, bold=False, parent='1', align='left'):
        return self.cell(x, y, w, h, text,
                         f'text;fillColor=none;strokeColor=none;fontSize={size};'
                         f'fontColor={color};fontStyle={1 if bold else 0};align={align};'
                         'verticalAlign=middle;spacing=0;', parent)

    def line(self, points, color=TEAL, arrow=True, dash=False, parent='1'):
        cid = str(self.n)
        self.n += 1
        c = ET.SubElement(self.root, 'mxCell', id=cid, value='',
                          style=f'html=1;rounded=1;orthogonalLoop=1;jettySize=auto;'
                          f'strokeColor={color};strokeWidth=2;endArrow={"block" if arrow else "none"};'
                          f'endSize=9;dashed={int(dash)};', edge='1', parent=parent)
        g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as':'geometry'})
        for point, kind in [(points[0], 'sourcePoint'), (points[-1], 'targetPoint')]:
            ET.SubElement(g, 'mxPoint', x=str(point[0]), y=str(point[1]), **{'as':kind})
        if len(points) > 2:
            a = ET.SubElement(g, 'Array', **{'as':'points'})
            for x, y in points[1:-1]:
                ET.SubElement(a, 'mxPoint', x=str(x), y=str(y))

    def ellipse(self, x, y, w, h, fill=ORANGE, parent='1', angle=-20, alpha=90):
        return self.cell(x, y, w, h, style=f'ellipse;fillColor={fill};strokeColor={fill};'
                         f'rotation={angle};opacity={alpha};', parent=parent)

    def save(self):
        path = ROOT / 'figures' / f'{self.name}.drawio'
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.xml)
        ET.ElementTree(self.xml).write(path, encoding='utf-8', xml_declaration=True)
        return path


def bits(d, x, y, parent, variant=0, pitch=16, selected=False):
    values = [1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0]
    if variant:
        for i in ([6] if variant == 1 else [1,3,4,7,9,11,15]):
            values[i] = 1 - values[i]
    for i, bit in enumerate(values):
        color = TEAL if bit else '#b8d8d8'
        if selected and i == 6:
            color = ORANGE
        d.box(x+i*pitch, y, pitch-3, 22, fill=color, stroke='none', parent=parent)


def cache_panel(d, x, y, w, h):
    p = d.group(x,y,w,h)
    d.box(0,0,w,h,fill='#f5f8f9',stroke='none',parent=p)
    d.text(22,16,w-44,36,'Match features',26,bold=True,parent=p)
    d.text(22,68,w-44,26,'Query · 16-bit LSH',21,parent=p)
    bits(d,22,106,p,pitch=(w-44)/16)
    d.line([(w/2,139),(w/2,166)],parent=p)
    d.text(22,168,w-44,26,'Semantic cache',21,bold=True,parent=p)
    pitch=(w-96)/16
    for row in range(3):
        yy=212+row*48
        d.box(14,yy-7,w-28,36,fill='#deeeed' if row==1 else '#ffffff',
              stroke=TEAL if row==1 else '#e1e8ea',parent=p)
        bits(d,22,yy,p,variant=1 if row==1 else 2,pitch=pitch,selected=row==1)
        d.text(w-65,yy-2,45,25,'d<sub>c</sub>' if row==1 else 'd<sub>j</sub>',21,
               ORANGE if row==1 else '#7a8d94',True,parent=p,align='center')
    d.text(20,h-52,w-40,34,'Hamming distance ≤ threshold',19,TEAL,True,parent=p,align='center')


def candidates_panel(d, x,y,w,h):
    p=d.group(x,y,w,h)
    d.box(0,0,w,h,fill='#f5f8f9',stroke='none',parent=p)
    d.text(22,16,w-44,36,'Select depth candidates',26,bold=True,parent=p)
    for row,(title,local) in enumerate([('Global set · D',False),('Cache hit · D/R',True),('Cache miss · D',False)]):
        yy=74+row*100
        d.text(22,yy,w-44,28,title,21,TEAL if local else INK,True,parent=p)
        pitch=(w-44)/16
        for i in range(16):
            color=TEAL if local and 6<=i<=9 else (GRAY if local else '#8faab3')
            if local and i==7: color=ORANGE
            d.box(22+i*pitch,yy+38,pitch-3,36,fill=color,stroke='none',parent=p)
        if local:
            xx=22+pitch*7+pitch/2-1.5
            d.line([(xx,yy+90),(xx,yy+76)],color=ORANGE,parent=p)
            d.text(xx+8,yy+77,50,22,'d<sub>c</sub>',18,ORANGE,True,parent=p)
    d.text(20,h-47,w-40,28,'Retain original depth values',19,TEAL,True,parent=p,align='center')


def prediction_panel(d,x,y,w,h,horizontal=False):
    p=d.group(x,y,w,h)
    d.box(0,0,w,h,fill='#f5f8f9',stroke='none',parent=p)
    d.text(22,16,w-44,36,'Run the depth predictor',26,bold=True,parent=p)
    labels=['Cross-view sampling','Cost volume','U-Net refinement','Depth estimate']
    if horizontal:
        for i,label in enumerate(labels):
            xx=22+i*(w-44)/4
            d.box(xx,78,(w-44)/4-16,72,label,fill=PALE if i<3 else '#dceae3',
                  stroke='none',parent=p,extra='fontSize=21;')
            if i<3:d.line([(xx+(w-44)/4-12,114),(xx+(w-44)/4-2,114)],parent=p)
    else:
        for i,label in enumerate(labels):
            yy=77+i*76
            d.box(80,yy,w-102,52,label,fill=PALE if i<3 else '#dceae3',
                  stroke='none',parent=p,extra='fontSize=21;')
            if i<3:d.line([(w/2+20,yy+54),(w/2+20,yy+73)],parent=p)
            if i==0:
                for a in range(3):
                    for b in range(3):d.box(25+b*12,yy+7+a*12,9,9,fill=TEAL if (a+b)%2 else '#9abdc3',stroke='none',parent=p)
            elif i==1:
                for j in range(4):d.box(24+j*6,yy+10-j*3,22,33,fill='#b9d6dc',stroke='#6d9ba7',parent=p)
            elif i==2:
                d.line([(25,yy+8),(25,yy+38),(57,yy+38),(57,yy+8)],arrow=False,parent=p)
                for xx,yyy in [(20,yy+2),(20,yy+32),(50,yy+2),(50,yy+32)]:d.box(xx,yyy,12,12,fill=TEAL,stroke='none',parent=p)
            else:
                for a in range(3):
                    for b in range(3):d.box(25+b*12,yy+7+a*12,11,11,fill=['#285d78','#4c8fa1','#b4d2c5'][b],stroke='none',parent=p)


def fsdr(mobile=False):
    name='fsdr-mobile' if mobile else 'fsdr'
    d=Diagram(name,700 if mobile else 1240,720 if mobile else 500)
    if mobile:
        cache_panel(d,10,10,320,430)
        candidates_panel(d,370,10,320,430)
        d.line([(333,270),(365,270)])
        prediction_panel(d,10,490,680,190,True)
        d.line([(530,442),(530,464),(350,464),(350,488)])
        d.text(12,686,676,28,'Write depth + signature back to the semantic cache',20,TEAL,True,align='center')
    else:
        cache_panel(d,10,10,360,410)
        candidates_panel(d,420,10,390,410)
        prediction_panel(d,860,10,370,410)
        d.line([(374,280),(395,280),(395,240),(414,240)])
        d.line([(814,220),(835,220),(835,113),(854,113)])
        d.line([(1140,423),(1140,468),(190,468),(190,423)],dash=True)
        d.box(390,445,510,45,'Write depth + signature back to the cache',fill='#ffffff',stroke='none',extra='fontSize=21;fontColor='+TEAL+';')
    return d.save()


def gaussians(d,x,y,w,h,parent,kind):
    if kind=='full':
        for r in range(4):
            for c in range(4):
                d.ellipse(x+c*w/4+2,y+r*h/4+4,w/5.4,h/7.5,parent=parent,angle=(-25+12*((r+c)%4)))
    else:
        for r in range(2):
            for c in range(2):
                cx=x+c*w/2;cy=y+r*h/2
                if kind=='weighted':
                    d.ellipse(cx-3,cy-3,w/2.1,h/2.1,fill='#e9bba0',parent=parent,angle=-18,alpha=35)
                    d.ellipse(cx+5,cy+5,w/3,h/3,fill=ORANGE,parent=parent,angle=-18)
                    d.ellipse(cx+w/5,cy+h/6,5,5,fill=INK,parent=parent,angle=0)
                else:d.ellipse(cx+2,cy+3,w/2.4,h/2.8,parent=parent,angle=-18)


def probes_panel(d,x,y,w,h,mobile=False):
    p=d.group(x,y,w,h)
    d.box(0,0,w,h,fill='#f5f8f9',stroke='none',parent=p)
    d.text(22,16,w-44,36,'Compute the probes first',26,bold=True,parent=p)
    from PIL import Image
    im=Image.open(ROOT/'media/scene-bedroom.webp')
    iw,ih=im.size
    im=im.crop((round(iw*.54),round(ih*.55),round(iw*.78),round(ih*.79))).resize((200,200))
    output=io.BytesIO();im.save(output,format='PNG')
    src=base64.b64encode(output.getvalue()).decode()
    sx,sy,size=(30,80,180) if mobile else (65,78,190)
    d.cell(sx,sy,size,size,style='shape=image;aspect=fixed;image=data:image/png,'+src+';',parent=p)
    step=size/4
    for i in range(1,4):
        d.line([(sx+i*step,sy),(sx+i*step,sy+size)],color='#ffffff',arrow=False,parent=p)
        d.line([(sx,sy+i*step),(sx+size,sy+i*step)],color='#ffffff',arrow=False,parent=p)
    for r,c in [(0,0),(0,3),(3,0),(3,3)]:
        d.ellipse(sx+(c+.5)*step-8,sy+(r+.5)*step-8,16,16,fill=ORANGE,parent=p,angle=0)
    d.text(sx,sy+size+5,size,28,'4 × 4 tile',20,parent=p,align='center')
    if mobile:
        d.line([(224,170),(273,170)],parent=p)
        d.box(285,95,w-310,65,'Original depth<br>+ Gaussian adaptor',fill=PALE,stroke='none',parent=p)
        d.line([(w-175,166),(w-175,195)],parent=p)
        gaussians(d,w-250,210,150,60,p,'representative')
        d.text(w-290,275,230,28,'Probe Gaussians',20,TEAL,True,parent=p,align='center')
    else:
        d.line([(160,306),(160,333)],parent=p)
        d.box(25,337,w-50,60,'Original depth +<br>Gaussian adaptor',fill=PALE,stroke='none',parent=p)
        d.line([(160,401),(160,423)],parent=p)
        gaussians(d,88,427,145,62,p,'representative')
        d.text(25,504,w-50,28,'Probe Gaussians',21,TEAL,True,parent=p,align='center')


def routes(d,x,y,w,h,mobile=False):
    p=d.group(x,y,w,h)
    d.text(0,0,w,36,'Select one path for the remaining pixels',26,bold=True,parent=p)
    if mobile:
        col1,col2,col3=160,210,130
        xx2,xx3=220,485
        ys=[85,255,425]
    else:
        col1,col2,col3=185,235,170
        xx2,xx3=245,540
        ys=[95,255,415]
    for i,(level,condition,operation,kind,color) in enumerate([
        ('L0','Low feature<br>variance?','Soft assignment<br>+ moment matching','representative',GREEN),
        ('L1','Low depth<br>spread?','Depth-weighted<br>aggregation','weighted',TEAL),
        ('Full','Irregular tile','S2 depth +<br>per-pixel adaptor','full',ORANGE)]):
        yy=ys[i]
        d.text(0,yy-30,col1,28,level,23,color,True,parent=p)
        d.box(0,yy,col1,82,condition,fill='#f1f6f5' if i<2 else '#fbf1eb',stroke=color,parent=p,extra='fontSize=22;')
        d.line([(col1+3,yy+41),(xx2-6,yy+41)],color=color,parent=p)
        if i<2:d.text(col1+3,yy+3,xx2-col1-5,25,'Yes',18,color,True,parent=p,align='center')
        d.box(xx2,yy,col2,82,operation,fill='#edf4f4' if i<2 else '#fbf1eb',stroke='none',parent=p,extra='fontSize=22;')
        d.line([(xx2+col2+5,yy+41),(xx3-8,yy+41)],color=color,parent=p)
        gaussians(d,xx3,yy+2,col3,80,p,kind)
        if i<2:
            d.line([(col1/2,yy+87),(col1/2,ys[i+1]-33)],color='#86979e',parent=p)
            d.text(col1/2+12,yy+91,45,26,'No',18,'#63767d',True,parent=p)
    d.text(xx3,40,col3,32,'Outputs',20,'#647b84',True,parent=p,align='center')


def saes(mobile=False):
    d=Diagram('saes-mobile' if mobile else 'saes',700 if mobile else 1140,930 if mobile else 570)
    if mobile:
        probes_panel(d,10,10,680,320,True)
        d.line([(350,335),(350,365)])
        routes(d,35,380,630,540,True)
    else:
        probes_panel(d,10,10,320,550)
        d.line([(334,463),(357,463),(357,158),(384,158)])
        routes(d,390,22,730,535)
    return d.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    paths=[fsdr(),fsdr(True),saes(),saes(True)]
    if args.export:
        for path in paths:
            for fmt,folder in [('svg',ROOT/'media'),('png',Path('/tmp/scarf-page-qa'))]:
                folder.mkdir(parents=True, exist_ok=True)
                cmd=['xvfb-run','-a','drawio','-x','-f',fmt,'-b','0','-o',str(folder/(path.stem+'.'+fmt)),str(path),'--disable-gpu']
                if fmt=='svg':cmd.insert(6,'-e')
                result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
                if result.returncode:raise RuntimeError(result.stdout)
                print(path.stem,fmt,'exported')
    else:
        for path in paths:print(path)
