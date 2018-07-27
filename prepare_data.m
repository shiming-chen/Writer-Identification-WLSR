clear;
p = '/home/lab535-user1/chenshm/Writer_Ident/data/cvl-database-1-1/cvl_train_patches256_new/';
pp = '/home/lab535-user1/chenshm/Writer_Ident/data/cvl-database-1-1/cvl_train_patches256_new/*.jpg';
imdb.meta.sets=['train','test'];
file = dir(pp);
counter_data=1;
counter_last = 1;
class = 0;
c_last = '';
cc = [];
for i=1:length(file)
    url = strcat(p,file(i).name);
    im = imread(url);
    %im = imresize(im,[256,256]); % save 256*256 image in advance for faster IO
    %url256 = strcat(pw,file(i).name);
    %imwrite(im,url256);
    imdb.images.data(counter_data) = cellstr(url); 
    c = strsplit(file(i).name,'-');
    if(~isequal(c{1},c_last))
        class = class + 1;
        fprintf('%d::%d\n',class,counter_data-counter_last);
        cc=[cc;counter_data-counter_last];
        c_last = c{1};
        counter_last = counter_data;
    end
    imdb.images.label(:,counter_data) = class;
    counter_data = counter_data + 1;
end
disp(i);
s = counter_data-1;
imdb.images.set = ones(1,s);
imdb.images.set(:,randi(s,[round(0.1*s),1])) = 2;

% no validation for small class 
cc = [cc;9];
cc = cc(2:end);
list = find(imdb.images.set==2);
for i=1:numel(list)
    if cc(imdb.images.label(list(i)))<10
        imdb.images.set(list(i))=1;
    end
end

save('cvl_train_patches256_train.mat','imdb','-v7.3');
