require 'cudnn'
require 'cunn'
require 'image'
local gm = assert(require 'graphicsmagick')
loadSize   = {3, 256, 256}
sampleSize = {3, 224, 224}
testHook = function(self, path)
   local oH = sampleSize[2]
   local oW = sampleSize[3];
   local out = torch.Tensor(10, 3, oW, oH)

   local input = gm.Image():load(path, loadSize[3], loadSize[2])
   -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
   local iW, iH = input:size()
   if iW < iH then
      input:size(256, 256 * iH / iW);
   else
      input:size(256 * iW / iH, 256);
   end
   iW, iH = input:size();
   local im = input:toTensor('float','RGB','DHW')
   -- mean/std
   mean = 0.028466778270914;
   std = 0.10084323727431;
   for i=1,3 do -- channels
      im[{{i},{},{}}]:add(-mean);
      im[{{i},{},{}}]:div(std);
   end

   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   out[1] = image.crop(im, w1, h1, w1+oW, h1+oW) -- center patch
   out[2] = image.hflip(out[1])
   h1 = 1; w1 = 1;
   out[3] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-left
   out[4] = image.hflip(out[3])
   h1 = 1; w1 = iW-oW;
   out[5] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-right
   out[6] = image.hflip(out[5])
   h1 = iH-oH; w1 = 1;
   out[7] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-left
   out[8] = image.hflip(out[7])
   h1 = iH-oH; w1 = iW-oW;
   out[9] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-right
   out[10] = image.hflip(out[9])

   return out
end
model = torch.load('model_3.t7')
model:evaluate()
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..directory..'"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end
files = scandir('test')

iii = 0;

repeat
iii = iii+1;
if iii%25==0 then
io.stderr:write("finished ")
io.stderr:write(iii)
io.stderr:write("\n")
end
outputt = "";
answer = io.read()
if answer=="end" then
else
prediction = model:forward(testHook({3,256,256},'test/'..answer)[1]:cuda())
outputt = outputt..answer..",";
for i = 1, 119 do
outputt = outputt..tostring(prediction[i])..",";
end
outputt = outputt.."0,0\n";
end
print(outputt)
until answer=="end"

