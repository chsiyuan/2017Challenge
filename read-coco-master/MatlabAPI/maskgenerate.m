coco = CocoApi( '/home/yl/Desktop/coco-master/annotations/instances_val2014.json' );
ids = getAnnIds( coco );
%idsim = getImgIds( coco);
anns = loadAnns( coco, ids );
t = coco.data.images;
boundingbox=[];
load matrix_select.mat

for i=1:13876
    %1:13876
    %14043

if i/200 == round (i/200)
    i
end

imid = M(i,1); 
%anns(i).image_id;
idxim = find([t.id]==imid);
height = t(idxim).height;
width = t(idxim).width;
%mask=zeros(height,width);
an1 = M(i,2);
cat1 = M(i,6);
an2 = M(i,3);
cat2= M(i,7);

seg1 = anns(an1).segmentation;
a = seg1{1};
l = length(a);
x = a(1:2:l);
%x = round(x);
y = a(2:2:l);
%y = round(y);
mask = poly2mask(x,y,height,width);
mask = double(mask);

len = length(seg1);
if len > 1
    for j=2:len
        a_tmp = seg1{j};
        l_tmp = length(a_tmp);
        x_tmp = a_tmp(1:2:l_tmp);
        %x = round(x);
        y_tmp = a_tmp(2:2:l_tmp);
        %y = round(y);
        mask_tmp = poly2mask(x_tmp,y_tmp,height,width);
        mask_tmp = double(mask_tmp);
        mask = max(mask, mask_tmp);
    end
    
end


mask2 = zeros(height,width);
if an2 ~= 0
    seg2 = anns(an2).segmentation;
    a2 = seg2{1};
    l2 = length(a2);
    x2 = a2(1:2:l2);
    %x = round(x);
    y2 = a2(2:2:l2);
    %y = round(y);
    mask2 = poly2mask(x2,y2,height,width);
    mask2 = double(mask2);
    len = length(seg2);
if len > 1
    for j=2:len
        a_tmp = seg2{j};
        l_tmp = length(a_tmp);
        x_tmp = a_tmp(1:2:l_tmp);
        %x = round(x);
        y_tmp = a_tmp(2:2:l_tmp);
        %y = round(y);
        mask_tmp = poly2mask(x_tmp,y_tmp,height,width);
        mask_tmp = double(mask_tmp);
        mask2 = max(mask2, mask_tmp);
    end
    
end
end



%generate grount truth
% !!!!! imshow
mask_gt = uint8(max(cat1*mask,cat2*mask2));
m_rlegt = MaskApi.encode(uint8(mask_gt));
bboxgt = MaskApi.toBbox( m_rlegt );
boundingbox=[boundingbox; imid bboxgt];

%merge 2 masks
mask_merge = uint8(255*(1-(1-mask).*(1-mask2)));
%imwrite(mask_merge,['mt',num2str(1),'.png']);
m_rle = MaskApi.encode(uint8(mask_merge));
%calculate bounding box
bbox = MaskApi.toBbox( m_rle );
clear m_rle;
clear m_rlegt;
clear cat1;
clear cat2;
% yi xia suo you bian huan dou yao bao zheng mask de size yu yuan tu yi zhi

%scalling: yi boundingbox wei zhong xin jin xing scaling bian huan. bian huan hou boundingbox de zhong xin zuo biao bu bian 
center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2];
scale = (rand(1)-0.5)/10;
H_s = [1+scale 0 0;0 1+scale 0;0 0 1];
H_s = [1 0 -scale*center(1);0 1 -scale*center(2);0 0 1]*H_s;
tf = maketform('projective',H_s');
mask_s = imtransform(mask_merge,tf,'XData',[1 size(mask_merge,2)],'YData',[1 size(mask_merge,1)]);
%mask_s = imcrop(mask_s,[center(1)*0.7-center(1) center(2)*0.7-center(2) width-1 height-1]);
clear tf;

%translation
x_scale = (rand(1)-0.5)/5;
y_scale = (rand(1)-0.5)/5;
H_t = [1 0 x_scale*bbox(3);0 1 y_scale*bbox(4);0 0 1];
tf = maketform('projective',H_t');
mask_t = imtransform(mask_s,tf,'XData',[1 size(mask_merge,2)],'YData',[1 size(mask_merge,1)]);
clear tf;
clear x_scale;
clear y_scale;
clear H_t;

%Thin-Plate Spline
ps = [90 90;90 height-89;width-89 height-89; width-89 90;round(width/2) round(height/2)];
pd = ps;
pd = [0.2*width*(rand(5,1)-0.5),0.2*height*(rand(5,1)-0.5)]+ps;
[mask_tps m_mask]=rbfwarp2d(mask_t,ps,pd,'thin');
mask_tps = uint8(255*round(mask_tps/255));
clear ps;
clear pd;
clear m_mask;
clear mask_s;
clear mask_t;

clear seg1;
clear seg2
clear a;
clear a2;
clear height;
clear width;
clear x;
clear x2;
clear y;
clear y2;
clear mask;
clear mask2;
clear l;
clear mask_merge;
clear center;
clear H_s;
clear scale;

%coarsening using dilation operation
se=strel('disk',5);
mask_c=imdilate(mask_tps,se);
imwrite(mask_gt,['/home/yl/Desktop/groundtruth/groundt_',num2str(imid),'.png']);
imwrite(mask_c,['/home/yl/Desktop/mask_def/deformation_',num2str(imid),'.png']);
clear mask_tps;
clear imid;
clear idxim;
clear se;
clear bbox;
clear bboxgt;
clear a_tmp;
clear l_tmp;
clear x_tmp;
clear y_tmp;
clear mask_tmp;
end
save boundingbox_matrix boundingbox;

