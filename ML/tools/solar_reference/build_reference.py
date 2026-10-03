"""Build the review-only reference example. Does not build any mass packets."""
from pathlib import Path
import hashlib, json, shutil, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ML=HERE.parents[1]
SOURCE=ML/'data/panel/20261003_220702_906/frames'
OUT=Path('/home/ccw100/Downloads/solar_test/reference_example')
FRAMES=list(range(800,1041,30))
UV=np.float64([(u,v) for v in (2,4,6,8) for u in (1,2,3)])
CLICKS=np.float64([[347,580],[525,545],[681,515],[350,765],[509,724],[645,687],
                   [352,911],[494,865],[616,823],[355,1027],[482,980],[590,935]])
MANUAL={
  890:[[1,2,554,1035],[2,2,817,1007],[1,4,534,1287],[2,4,766,1250],[3,4,971,1218],[0,2,284,1069],[0,4,304,1320]],
  920:[[0,1,249,875],[1,1,805,865],[0,2,280,1122],[1,2,759,1107],[0,3,306,1318],[1,3,715,1297]],
  950:[[0,1,154,850],[1,1,774,834],[0,2,216,1125],[1,2,738,1106],[0,3,265,1314],[1,3,710,1293]],
  980:[[0,1,138,922],[1,1,767,934],[0,2,177,1209],[1,2,711,1208],[0,3,208,1398],[1,3,670,1398]],
  1010:[[0,1,132,962],[1,1,826,954],[0,2,188,1278],[1,2,771,1267]],
}
# Estimated intersections of the extended cell edges at chamfered top corners.
# These constrain the plane fit but are NOT counted as directly observed points.
FIT_ONLY={1010:[[0,0,54,509],[1,0,887,523]]}

def project(H,uv):
    return cv2.perspectiveTransform(np.asarray(uv,dtype=np.float64).reshape(1,-1,2),H)[0]

def refine_diamonds(im,H):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    predicted=project(H,UV); out=[]
    for uv,p in zip(UV,predicted):
        x,y=p
        if not (15<x<im.shape[1]-15 and 15<y<im.shape[0]-15): continue
        axes=project(H,[uv,uv+[1,0],uv+[0,1]])
        scale=min(np.linalg.norm(axes[1]-axes[0]),np.linalg.norm(axes[2]-axes[0]))
        radius=int(np.clip(scale*.14,12,50))
        x0,y0=max(0,int(x)-radius),max(0,int(y)-radius)
        x1,y1=min(im.shape[1],int(x)+radius+1),min(im.shape[0],int(y)+radius+1)
        patch=gray[y0:y1,x0:x1]
        threshold=float(np.percentile(patch,40)+.55*(np.percentile(patch,95)-np.percentile(patch,40)))
        mask=(patch>threshold).astype('uint8')
        dist=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        # The diamond's broad white center is thicker than the narrow gaps/busbars.
        yy,xx=np.mgrid[:patch.shape[0],:patch.shape[1]]
        score=dist*np.exp(-((xx-(x-x0))**2+(yy-(y-y0))**2)/(2*(radius*.65)**2))
        iy,ix=np.unravel_index(np.argmax(score),score.shape)
        if dist[iy,ix]<2.0: continue
        weights=np.maximum(dist-dist[iy,ix]*.8,0)
        weights*=((xx-ix)**2+(yy-iy)**2<max(3,dist[iy,ix])**2)
        if weights.sum():
            px=x0+float((xx*weights).sum()/weights.sum())
            py=y0+float((yy*weights).sum()/weights.sum())
            if np.linalg.norm(np.array([px,py])-p)<radius*.75:
                out.append((uv.copy(),[px,py],float(dist[iy,ix])))
    return out

def track(previous,current,H):
    gray0=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY);gray1=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    mask=np.zeros(gray0.shape,np.uint8)
    q=project(H,[[0,0],[4,0],[4,9],[0,9]])
    cv2.fillConvexPoly(mask,np.round(q).astype('int32'),255)
    mask=cv2.erode(mask,np.ones((9,9),np.uint8))
    p=cv2.goodFeaturesToTrack(gray0,maxCorners=500,qualityLevel=.015,minDistance=9,mask=mask,blockSize=7)
    if p is None or len(p)<12: raise RuntimeError('Insufficient panel features')
    args=dict(winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,ok,_=cv2.calcOpticalFlowPyrLK(gray0,gray1,p,None,**args)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,q,None,**args)
    good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(p-back,axis=2).ravel()<.7)
    if good.sum()<12: raise RuntimeError('Insufficient consistent tracks')
    M,inlier=cv2.findHomography(p[good],q[good],cv2.RANSAC,1.5)
    if M is None or inlier.sum()<10: raise RuntimeError('Homography tracking failed')
    err=np.linalg.norm(project(M,p[good].reshape(-1,2))-q[good].reshape(-1,2),axis=1)
    return M@H,dict(features=int(good.sum()),inliers=int(inlier.sum()),median_fit_residual_px=float(np.median(err[inlier.ravel()>0])))

def render(im,H,points,name):
    vis=im.copy(); h,w=vis.shape[:2]
    if H is None:
        cv2.rectangle(vis,(0,0),(w,105),(20,20,20),-1)
        cv2.putText(vis,f'{name} | close-up: no reliable grid junctions',(15,30),0,.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Identity retained; uncertain point coordinates = null',(15,60),0,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'No projected grid is asserted for this blurred close-up',(15,89),0,.62,(200,200,255),1,cv2.LINE_AA)
        cv2.putText(vis,'C0 R0 (identity from preceding frames)',(100,650),0,.7,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(vis,'C0 R0 (identity from preceding frames)',(100,650),0,.7,(255,255,255),1,cv2.LINE_AA)
        return vis
    # Clip finite projected line segments; never paint onto source JPEGs.
    for uv in [[(u,0),(u,9)] for u in range(5)]+[[(0,v),(4,v)] for v in range(10)]:
        a,b=project(H,uv)
        if np.isfinite([a,b]).all() and np.max(np.abs([a,b]))<100000:
            ok,p0,p1=cv2.clipLine((0,0,w,h),tuple(np.round(a).astype(int)),tuple(np.round(b).astype(int)))
            if ok: cv2.line(vis,p0,p1,(255,200,0),2,cv2.LINE_AA)
    for p in points:
        if p['visibility']!='visible':continue
        x,y=int(round(p['x_px'])),int(round(p['y_px']))
        color=(0,255,80) if p['method'] in ['image_refined_diamond','manual_intersection'] else (0,190,255)
        cv2.circle(vis,(x,y),6,color,2,cv2.LINE_AA)
        label=f"({p['u']},{p['v']})"
        tx=max(2,min(w-92,x+9));ty=max(18,min(h-7,y-9))
        cv2.putText(vis,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(vis,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    for v in range(9):
        for u in range(4):
            x,y=project(H,[[u+.5,v+.5]])[0]
            if 25<x<w-100 and 35<y<h-20:
                cv2.putText(vis,f'C{u} R{v}',(int(x)-30,int(y)),0,.52,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(vis,f'C{u} R{v}',(int(x)-30,int(y)),0,.52,(255,255,255),1,cv2.LINE_AA)
    cv2.rectangle(vis,(0,0),(w,75),(20,20,20),-1)
    cv2.putText(vis,f'{name} | labels = (u,v); cells = Ccol Rrow',(15,29),0,.66,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(vis,'GREEN: observed point | AMBER/CYAN: projected grid estimate',(15,58),0,.57,(210,240,210),1,cv2.LINE_AA)
    return vis

def main():
    for sub in ['images','visual_key']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    previous=cv2.imread(str(SOURCE/'00800.jpg'))
    H,_=cv2.findHomography(UV,CLICKS)
    initial=refine_diamonds(previous,H)
    H,_=cv2.findHomography(np.array([p[0] for p in initial]),np.array([p[1] for p in initial]),method=0)
    manifests=[];annotations=[];tiles=[];diagnostics=[]
    for n in range(FRAMES[0],FRAMES[-1]+1):
        current=cv2.imread(str(SOURCE/f'{n:05d}.jpg'))
        if n>FRAMES[0]: H,diag=track(previous,current,H)
        else: diag={'features':None,'inliers':None,'median_fit_residual_px':None}
        previous=current
        if n not in FRAMES: continue
        if n in MANUAL:
            a=np.float64(MANUAL[n]+FIT_ONLY.get(n,[]));H,_=cv2.findHomography(a[:,:2],a[:,2:],method=0)
            diag['manual_anchor_fit_rmse_px']=float(np.sqrt(np.mean(np.sum((project(H,a[:,:2])-a[:,2:])**2,axis=1))))
        refined=refine_diamonds(current,H)
        # Re-anchor to observed diamond centers if they span two dimensions.
        if len(refined)>=4 and len(set(int(x[0][0]) for x in refined))>=2 and len(set(int(x[0][1]) for x in refined))>=2:
            proposed,ins=cv2.findHomography(np.array([p[0] for p in refined]),np.array([p[1] for p in refined]),cv2.RANSAC,3)
            if proposed is not None and ins.sum()>=4:
                shift=np.median(np.linalg.norm(project(proposed,[p[0] for p in refined])-project(H,[p[0] for p in refined]),axis=1))
                if shift<15:H=proposed
        observed={tuple(int(x) for x in uv):xy for uv,xy,_ in refined}
        rows=[]
        for v in range(10):
            for u in range(5):
                xy=project(H,[[u,v]])[0]
                method='tracked_homography'
                manual={(int(p[0]),int(p[1])):p[2:] for p in MANUAL.get(n,[])}
                if (u,v) in manual:xy=np.array(manual[(u,v)]);method='manual_intersection'
                if (u,v) in observed:xy=np.array(observed[(u,v)]);method='image_refined_diamond'
                visible=bool(0<=xy[0]<1080 and 0<=xy[1]<1440)
                rows.append(dict(track_id=f'panel0:u{u}:v{v}',panel_id='panel0',u=u,v=v,
                                 x_px=round(float(xy[0]),3) if visible else None,
                                 y_px=round(float(xy[1]),3) if visible else None,
                                 visibility='visible' if visible else 'out_of_frame',method=method))
        seq=len(manifests);filename=f'{seq+1:06d}_frame_{n:05d}.jpg';image_file='images/'+filename
        src=SOURCE/f'{n:05d}.jpg';shutil.copy2(src,OUT/image_file)
        frame_id=f'20261003_220702_906:{n:05d}'
        manifests.append(dict(sequence_index=seq,frame_id=frame_id,source_frame=n,image_file=image_file,width=1080,height=1440,
                              timestamp_seconds=round((n-1)*1001/30000,6),example_elapsed_seconds=round((n-FRAMES[0])*1001/30000,6),
                              sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        confident=n!=1040
        if not confident:
            for row in rows:
                row.update(x_px=None,y_px=None,visibility='uncertain',method='inferred')
        annotation=dict(frame_id=frame_id,image_file=image_file,status='tracking' if confident else 'uncertain',points=rows,
                        homography_panel_to_image=(H/H[2,2]).tolist() if confident else None,quality=diag)
        if not confident:
            annotation['notes']='Blurred close-up with no reliably measurable grid junctions. C0R0/C0R1 identity follows earlier frames; abstain from grid coordinates rather than extrapolate unreliable geometry.'
        annotations.append(annotation)
        vis=render(current,H if confident else None,rows,f'Frame {n:05d}')
        cv2.imwrite(str(OUT/'visual_key'/filename),vis,[cv2.IMWRITE_JPEG_QUALITY,95])
        tiles.append(cv2.resize(vis,(360,480)))
        diagnostics.append(dict(frame=n,observed_diamonds=len(observed),**diag))
        print(n,len(observed),diag,flush=True)
    doc=dict(schema_version=1,sequence_id='solar_zoom_reference_v1',recording_id='20261003_220702_906',
             coordinate_convention='Original 1080x1440 image pixels: x right, y down, origin top-left; u column boundary 0..4, v row boundary 0..9.',
             panel_spec=dict(cols=4,rows=9,diamond_rows=[2,4,6,8]),
             label_provenance='Manual grid numbering and initial point seeds; adjacent-frame optical-flow homographies; local image refinement of visible diamond centers; visual overlay review. Projected intersections are estimates, not direct pixel measurements. Not human-certified ground truth.',
             frames=annotations)
    manifest=dict(schema_version=1,sequence_id=doc['sequence_id'],recording_id=doc['recording_id'],
                  sampling=dict(first_source_frame=800,last_source_frame=1040,stride=30,nominal_source_fps='30000/1001',
                                timestamp_provenance='Nominal CFR extracted-image timeline; not exact original video PTS.',duration_seconds=8.008),frames=manifests)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'tracking.json').write_text(json.dumps(doc,indent=2)+'\n')
    template={**doc,'label_provenance':'Unlabeled template; replace with model output.',
              'frames':[dict(frame_id=f['frame_id'],image_file=f['image_file'],status='unlabeled',points=[]) for f in annotations]}
    (OUT/'tracking_template.json').write_text(json.dumps(template,indent=2)+'\n')
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    cv2.imwrite(str(OUT/'preview.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)]))

if __name__=='__main__': main()
