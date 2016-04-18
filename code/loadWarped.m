function Is = loadWarped(filepath, refid, startid, endid)
%%
if ~exist('refid', 'var')
    refid = 60;
end
if ~exist('startid', 'var') || ~exist('endid', 'var')
    startid = 30;
    endid = 89;
end
assert(startid < endid);
tempImg = imread(sprintf('%s/temp/warpedb%05d_%05d.jpg', filepath, refid, startid));
[height,width, ~] = size(tempImg);
Is = zeros(height, width, endid-startid+1);
for i=startid:endid
    path = sprintf('%s/temp/warpedb%05d_%05d.jpg', ...
        filepath, refid, i);
    disp(path);
    Is(:,:,i-startid+1) = imgaussfilt(rgb2gray(im2double(imread(path))), 3);


end