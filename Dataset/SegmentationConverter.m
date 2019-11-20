filePath = './ade20k';
outPath = './Converted';

S = dir(fullfile(filePath,'*_seg.png')); % pattern to match filenames.
segFiles = [];
fileName = [];
for k = 1:numel(S)
    F = S(k).name;
    readFileName = filePath + "/" + string(F);
    I = imread(readFileName);
    fileoutname = outPath + "/" + string(F);
    convertFromADE(I, fileoutname);
end