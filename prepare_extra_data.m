load('./data/cvl_train_patches256_new.mat');
%p = dir('/home/lab535-user1/chenshm/Writer_Ident/data/cvl-database-1-1/CVL_IAM_g06_binary/*.jpg');
%p = dir('/home/lab535-user1/chenshm/Writer_Ident/CFWI/first_stage/cvl_words/generated_245760/*.jpg');
p = dir('//home/lab535-user1/chenshm/Writer_Ident/data/ICDAR2013/icdar2013_test_Line/icdar2013_test_all_Line_patches256_new/*.jpg');


num = numel(imdb.images.data);
%for i=1:length(p) %numel(p)
for i=1:36000
    url = strcat('/home/lab535-user1/chenshm/Writer_Ident/data/ICDAR2013/icdar2013_test_Line/icdar2013_test_all_Line_patches256_new/',p(i).name);
    %url = strcat('/home/zzd/CUHK03/zzd_code/split1_256/',p(i).name);
    imdb.images.data(num+i) =cellstr(url);
    imdb.images.label(num+i) = 0;
    imdb.images.set(num+i) = 1;
end

save('words_patches256_ICDARx36000.mat','imdb','-v7.3');