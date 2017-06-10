load matrix_select.mat
for i=1:13876
imid = M(i,1); 
if imid<100
    im = imread(['/media/yl/My Passport/coco/val2014/COCO_val2014_0000000000',num2str(imid),'.jpg']);
    imwrite(im,['/home/yl/Desktop/images/COCO_val2014_0000000000',num2str(imid),'.jpg']);
elseif imid<1000
    im = imread(['/media/yl/My Passport/coco/val2014/COCO_val2014_000000000',num2str(imid),'.jpg']);
    imwrite(im,['/home/yl/Desktop/images/COCO_val2014_000000000',num2str(imid),'.jpg']);
elseif imid<10000
    im = imread(['/media/yl/My Passport/coco/val2014/COCO_val2014_00000000',num2str(imid),'.jpg']);
    imwrite(im,['/home/yl/Desktop/images/COCO_val2014_00000000',num2str(imid),'.jpg']);
elseif imid<100000
    im = imread(['/media/yl/My Passport/coco/val2014/COCO_val2014_0000000',num2str(imid),'.jpg']);
    imwrite(im,['/home/yl/Desktop/images/COCO_val2014_0000000',num2str(imid),'.jpg']);
else
    im = imread(['/media/yl/My Passport/coco/val2014/COCO_val2014_000000',num2str(imid),'.jpg']);
    imwrite(im,['/home/yl/Desktop/images/COCO_val2014_000000',num2str(imid),'.jpg']);
end
clear im;
clear imid;
if i/200 == round(i/200)
    i
end

end
