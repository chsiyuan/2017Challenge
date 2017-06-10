coco = CocoApi( '/home/yl/Desktop/coco-master/annotations/instances_val2014.json' );
ids = getAnnIds( coco );
%idsim = getImgIds( coco);
anns = loadAnns( coco, ids );
clear coco;
clear ids;
%t = coco.data.images;
category_vector = zeros(90,1);

%t = 0;
%imid_previous = anns(1).image_id;
S=[anns(1).image_id 1 0 anns(1).area 0 anns(1).category_id 0];


for i=2:200000
    if i/2000 == uint8(i/2000)
        i
    end
v = [anns(i).image_id anns(i).area anns(i).category_id];    

    if v(1) == S(end,1)
        if v(2) > S(end,4)
          S(end,3) = S(end,2);
          S(end,2) = i;
          S(end,5) = S(end,4);
          S(end,4) = v(2);
          S(end,7) = S(end,6);
          S(end,6) = v(3);
        elseif v(2) > S(end,5);
            S(end,3) = i;
            S(end,5) = v(2);
            S(end,7) = v(3);
        end
    else
        if ((S(end,6) == 1) & (S(end,7) == 1))
            if rand(1)>0.2
                S(end,:) = [];
            end
        end
        
        if ((S(end,6) > 43) || (S(end,7) > 43))
            if rand(1) > 0.033
            S(end,:) = [];
            end
        end
        t1 = S(end, 6);
        t2 = S(end, 7);
        if t1>0
            category_vector(t1) = category_vector(t1) + 1;
        end
        if t2>0
            category_vector(t2) = category_vector(t2) + 1;
        end
        
        S=[S; anns(i).image_id i 0 anns(i).area 0 anns(i).category_id 0];
    end

end

save select_matrix S;