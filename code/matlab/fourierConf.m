function conf = fourierConf(path,anchor,startid,endid,alpha,beta,tf)
%%
temp = imread(sprintf('%s/prewarpb%05d_%05d.jpg',path, anchor, startid));
[h,w,c] = size(temp);
N = endid - startid + 1;
Is = zeros(N,h,w,c);
disp('loading...');
for i=startid:endid
    Is(i-startid+1,:,:,:)=imread(sprintf('%s/prewarpb%05d_%05d.jpg',path, anchor, i));
end

conf = zeros(h,w);
disp('Computing confidence...');
min_colordiff=30;
for y=1:h
    for x=1:w
        seq = reshape(Is(:,y,x,:), N, c);
        %ignore small 
        if max(max(seq)-min(seq)) < min_colordiff
            conf(y,x) = 0.0;
            continue;
        end
        m = mean(seq);
        seq=seq-repmat(m,N,1);
        mag = abs(fft(seq));
        mag(ceil(N/2):end,:)=0;
        peak = max(mag(tf+1:end,:));
        ratio1 = median(peak ./ max(mag(1:tf,:)));
        %ratio2 = median(peak ./ median(mag));
        conf(y,x) = 1/(1+exp(-1*alpha*(ratio1-beta)));% * 1/(1+exp(-1*alpha*(ratio2-beta)));
        if x==-1 && y==-1
            disp('Debug...');
            disp(seq);
            figure; hold on;
            subplot(1,2,1);
            plot(seq);
            subplot(1,2,2);
            plot(mag);
            disp(max(mag(tf+ceil(1:N/2),:)) ./ max(mag(1:tf,:)));
            fprintf('median ratio: %.2f, conf: %.2f\n', ratio1, conf(y,x));
            hold off;
        end
    end
end

figure;
fg1 = subplot(1,2,1);
imshow(reshape(Is(end,:,:,:),h,w,c)/256);
fg2 = subplot(1,2,2);
imshow(conf);
linkaxes([fg1 fg2], 'xy');

end