coco = CocoApi( '/home/yl/Desktop/coco-master/annotations/instances_val2014.json' );
ids = getAnnIds( coco );
%idsim = getImgIds( coco);
anns = loadAnns( coco, ids );
boundingbox=[];
load select_matrix.mat
M=[]

for i=1:14490
imid = S(i,1); 
an1 = S(i,2);
an2 = S(i,3);
if an2 == 0
    an2 = an1;
end
a = class(anns(an1).segmentation);
b = class(anns(an2).segmentation);
if (length(a) ~= 6) & (length(b) ~= 6)
    M = [M; S(i,:)];
end

end

save matrix_select M;


