filePath = './ade20k';
outPath = './Converted';

S = dir(fullfile(filePath,'*_seg.png')); % pattern to match filenames.
segFiles = [];
fileName = [];
for k = 1:numel(S)
    F = S(k).name;
    readFileName = filePath + "/" + string(F);
    I = imread(readFileName);
    fileName{k} = F;
    segFiles{k} = I;
end

for i = 1:length(segFiles)
    fileoutname = outPath + "/" + fileName(i);
    convertFromADE(segFiles(i), fileoutname);
end