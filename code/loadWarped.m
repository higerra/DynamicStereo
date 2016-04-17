function Is = loadWarped(filepath, refid, startid, endid)
%%
if ~exist('refid', 'var')
    refid = 60;
end
if ~exist('startid', 'var') || ~exist('endid', 'var')
    startid = 30;
    endid = 60;
end
assert(startid < endid);
tempImg = imread(sprintf('%s/temp/warpedb%05d_%05d.jpg', filepath, refid, startid));
[height,width, ~] = size(tempImg);
Is = zeros(height, width, endid-startid+1);
for i=startid:endid
    Is(:,:,i-startid+1) = rgb2gray(imread(sprintf('%s/temp/warppedb%05d_%05d.jpg', ...
        filepath, refid, i)));
end

end