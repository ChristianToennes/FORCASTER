import numpy as np

def adapthisteq(varargin):
    #%ADAPTHISTEQ Contrast-limited Adaptive Histogram Equalization (CLAHE).
    #%   ADAPTHISTEQ enhances the contrast of images by transforming the
    #%   values in the intensity image I.  Unlike HISTEQ, it operates on small
    #%   data regions (tiles), rather than the entire image. Each tile's
    #%   contrast is enhanced, so that the histogram of the output region
    #%   approximately matches the specified histogram. The neighboring tiles
    #%   are then combined using bilinear interpolation in order to eliminate
    #%   artificially induced boundaries.  The contrast, especially
    #%   in homogeneous areas, can be limited in order to avoid amplifying the
    #%   noise which might be present in the image.
    #%
    #%   J = ADAPTHISTEQ(I) Performs CLAHE on the intensity image I.
    #%
    #%   J = ADAPTHISTEQ(I,PARAM1,VAL1,PARAM2,VAL2...) sets various parameters.
    #%   Parameter names can be abbreviated, and case does not matter. Each
    #%   parameter is followed by a value as indicated below:
    #%
    #%   'NumTiles'     Two-element vector of positive integers: [M N].
    #%                  [M N] specifies the number of tile rows and
    #%                  columns.  Both M and N must be at least 2.
    #%                  The total number of image tiles is equal to M*N.
    #%
    #%                  Default: [8 8].
    #%
    #%   'ClipLimit'    Real scalar from 0 to 1.
    #%                  'ClipLimit' limits contrast enhancement. Higher numbers
    #%                  result in more contrast.
    #%
    #%                  Default: 0.01.
    #%
    #%   'NBins'        Positive integer scalar.
    #%                  Sets number of bins for the histogram used in building a
    #%                  contrast enhancing transformation. Higher values result
    #%                  in greater dynamic range at the cost of slower processing
    #%                  speed.
    #%
    #%                  Default: 256.
    #%
    #%   'Range'        One of the strings or character vectors: 'original' or
    #%                  'full'.
    #%                  Controls the range of the output image data. If 'Range'
    #%                  is set to 'original', the range is limited to
    #%                  [min(I(:)) max(I(:))]. Otherwise, by default, or when
    #%                  'Range' is set to 'full', the full range of the output
    #%                  image class is used (e.g. [0 255] for uint8).
    #%
    #%                  Default: 'full'.
    #%
    #%   'Distribution' Distribution can be one of three strings or character
    #%                  vectors: 'uniform', 'rayleigh', 'exponential'.
    #%                  Sets desired histogram shape for the image tiles, by
    #%                  specifying a distribution type.
    #%
    #%                  Default: 'uniform'.
    #%
    #%   'Alpha'        Nonnegative real scalar.
    #%                  'Alpha' is a distribution parameter, which can be supplied
    #%                  when 'Dist' is set to either 'rayleigh' or 'exponential'.
    #%
    #%                  Default: 0.4.
    #%
    #%   Notes
    #%   -----
    #%   - 'NumTiles' specify the number of rectangular contextual regions (tiles)
    #%     into which the image is divided. The contrast transform function is
    #%     calculated for each of these regions individually. The optimal number of
    #%     tiles dep#ends on the type of the input image, and it is best determined
    #%     through experimentation.
    #%
    #%   - The 'ClipLimit' is a contrast factor that prevents over-saturation of the
    #%     image specifically in homogeneous areas.  These areas are characterized
    #%     by a high peak in the histogram of the particular image tile due to many
    #%     pixels falling inside the same gray level range. Without the clip limit,
    #%     the adaptive histogram equalization technique could produce results that,
    #%     in some cases, are worse than the original image.
    #%
    #%   - ADAPTHISTEQ can use Uniform, Rayleigh, or Exponential distribution as
    #%     the basis for creating the contrast transform function. The distribution
    #%     that should be used dep#ends on the type of the input image.
    #%     For example, underwater imagery appears to look more natural when the
    #%     Rayleigh distribution is used.
    #%
    #%   Class Support
    #%   -------------
    #%   Intensity image I can be uint8, uint16, int16, double, or single.
    #%   The output image J has the same class as I.
    #%
    #%   Example 1
    #%   ---------
    #%   #% Apply Contrast-Limited Adaptive Histogram Equalization to an
    #%   #% image and display the results.
    #%
    #%      I = imread('tire.tif');
    #%      A = adapthisteq(I,'clipLimit',0.02,'Distribution','rayleigh');
    #%      figure
    #%      montage([I,A]);
    #%
    #%   Example 2
    #%   ---------
    #%   #% Apply Contrast-Limited Adaptive Histogram Equalization to a color
    #%   #% photograph.
    #%
    #%      [X MAP] = imread('shadow.tif');
    #%      RGB = ind2rgb(X,MAP); #% convert indexed image to truecolor format
    #%      cform2lab = makecform('srgb2lab');
    #%      LAB = applycform(RGB, cform2lab); #%convert image to L*a*b color space
    #%      L = LAB(:,:,1)/100; #% scale the values to range from 0 to 1
    #%      LAB(:,:,1) = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005)*100;
    #%      cform2srgb = makecform('lab2srgb');
    #%      J = applycform(LAB, cform2srgb); #%convert back to RGB
    #%      figure
    #%      montage([RGB, J]);
    #%
    #%   See also HISTEQ, IMHISTMATCH.

    #%   Copyright 1993-2020 The MathWorks, Inc.

    #%   References:
    #%      Karel Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
    #%      Graphics Gems IV, p. 474-485, code: p. 479-484
    #%
    #%      Hanumant Singh, Woods Hole Oceanographic Institution, personal
    #%      communication

    #%--------------------------- The algorithm ----------------------------------
    #%
    #%  1. Obtain all the inputs:
    #%    * image
    #%    * number of regions in row and column directions
    #%    * number of bins for the histograms used in building image transform
    #%      function (dynamic range)
    #%    * clip limit for contrast limiting (normalized from 0 to 1)
    #%    * other miscellaneous options
    #%  2. Pre-process the inputs:
    #%    * determine real clip limit from the normalized value
    #%    * if necessary, pad the image before splitting it into regions
    #%  3. Process each contextual region (tile) thus producing gray level mappings
    #%    * extract a single image region
    #%    * make a histogram for this region using the specified number of bins
    #%    * clip the histogram using clip limit
    #%    * create a mapping (transformation function) for this region
    #%  4. Interpolate gray level mappings in order to assemble final CLAHE image
    #%    * extract cluster of four neighboring mapping functions
    #%    * process image region partly overlapping each of the mapping tiles
    #%    * extract a single pixel, apply four mappings to that pixel, and
    #%      interpolate between the results to obtain the output pixel; repeat
    #%      over the entire image
    #%
    #%  See code for further details.
    #%
    #%-----------------------------------------------------------------------------

    #matlab.images.internal.errorIfgpuArray(varargin[:]);

    [I, selectedRange, fullRange, numTiles, dimTile, clipLimit, numBins, noPadRect, distribution, alpha, int16ClassChange] = parseInputs(varargin)

    tileMappings = makeTileMappings(I, numTiles, dimTile, numBins, clipLimit, selectedRange, fullRange, distribution, alpha)

    #%Synthesize the output image based on the individual tile mappings.
    out = makeClaheImage(I, tileMappings, numTiles, selectedRange, numBins, dimTile)

    if int16ClassChange:
        #% Change uint16 back to int16 so output has same class as input.
        #out = images.internal.builtins.uint16toint16(out)
        out = np.array(out, dtype=int)
    #end

    if not noPadRect==0: #%do we need to remove padding?
        out = out[noPadRect.ulRow:noPadRect.lrRow, noPadRect.ulCol:noPadRect.lrCol]
    #end

    #%-----------------------------------------------------------------------------

def makeTileMappings(I, numTiles, dimTile, numBins, clipLimit, selectedRange, fullRange, distribution, alpha):

    numPixInTile = np.prod(dimTile)

    tileMappings = np.zeros(numTiles)

    #% extract and process each tile
    imgCol = 0
    for col in range(numTiles[1]):
        imgRow = 0;
        for row in range(numTiles[0]):

            tile = I[imgRow:imgRow+dimTile[0]-1,imgCol:imgCol+dimTile[1]-1]

            #% for speed, call MEX file directly thus avoiding costly
            #% input parsing of imhist
            #tileHist = images.internal.builtins.imhistc(tile, numBins, 1, fullRange[1]);
            tileHist = np.histogram(tile, numBins, fullRange)

            tileHist = clipHistogram(tileHist, clipLimit, numBins)

            tileMapping = makeMapping(tileHist, selectedRange, fullRange, numPixInTile, distribution, alpha)

            #% assemble individual tile mappings by storing them in a cell array;
            tileMappings[row,col] = tileMapping

            imgRow = imgRow + dimTile[0]
        #end
        imgCol = imgCol + dimTile[1] #% move to the next column of tiles
    #end
    return tileMappings

#%-----------------------------------------------------------------------------
#% Calculate the equalized lookup table (mapping) based on cumulating the input
#% histogram.  Note: lookup table is rescaled in the selectedRange [Min..Max].
eps = 1e-16
def makeMapping(imgHist, selectedRange, fullRange, numPixInTile, distribution, alpha):

    histSum = np.cumsum(imgHist)
    valSpread  = selectedRange[1] - selectedRange[0]

    if distribution=='uniform':
        scale =  valSpread/numPixInTile
        mapping = np.min(selectedRange[0] + histSum*scale, selectedRange[1]) #%limit to max

    elif distribution=='rayleigh': #% suitable for underwater imagery
        #% pdf = (t./alpha^2).*exp(-t.^2/(2*alpha^2))*U(t)
        #% cdf = 1-exp(-t.^2./(2*alpha^2))
        hconst = 2*alpha**2
        vmax = 1 - np.exp(-1/hconst)
        val = vmax*(histSum/numPixInTile)
        val[val>=1] = 1-eps #% avoid log(0)
        temp = np.sqrt(-hconst*np.log(1-val))
        mapping = min(selectedRange[0]+temp*valSpread, selectedRange[1]) #%limit to max

    elif distribution=='exponential':
        #% pdf = alpha*exp(-alpha*t)*U(t)
        #% cdf = 1-exp(-alpha*t)
        vmax = 1 - np.exp(-alpha)
        val = (vmax*histSum/numPixInTile)
        val[val>=1] = 1-eps
        temp = -1/alpha*np.log(1-val)
        mapping = min(selectedRange[0]+temp*valSpread, selectedRange[1]);

    else:
        raise('images:adapthisteq:distributionType') #%should never get here

    #end

    #%rescale the result to be between 0 and 1 for later use by the GRAYXFORMMEX
    #%private mex function
    mapping = mapping/fullRange[1]
    return mapping

#%-----------------------------------------------------------------------------
#% This function clips the histogram according to the clipLimit and
#% redistributes clipped pixels across bins below the clipLimit

def clipHistogram(imgHist, clipLimit, numBins):

    #% total number of pixels overflowing clip limit in each bin
    #totalExcess = sum(max(imgHist - clipLimit,0))
    totalExcess = imgHist - clipLimit
    totalExcess[totalExcess<0] = 0
    totalExcess = np.sum(totalExcess)

    #% clip the histogram and redistribute the excess pixels in each bin
    avgBinIncr = np.floor(totalExcess/numBins)
    upperLimit = clipLimit - avgBinIncr #% bins larger than this will be
                                        #% set to clipLimit

    #% this loop should speed up the operation by putting multiple pixels
    #% into the "obvious" places first
    for k in range(numBins):
        if imgHist[k] > clipLimit:
            imgHist[k] = clipLimit
        else:
            if imgHist[k] > upperLimit: #% high bin count
                totalExcess = totalExcess - (clipLimit - imgHist[k])
                imgHist[k] = clipLimit
            else:
                totalExcess = totalExcess - avgBinIncr
                imgHist[k] = imgHist[k] + avgBinIncr
            #end
        #end
    #end

    #% this loops redistributes the remaining pixels, one pixel at a time
    k = 0
    while (totalExcess != 0):
        #%keep increasing the step as fewer and fewer pixels remain for
        #%the redistribution (spread them evenly)
        #stepSize = max(floor(numBins/totalExcess),1);
        stepSize = max(np.floor(numBins/totalExcess), 1)
        for m in range(k, numBins, stepSize):
            if imgHist[m] < clipLimit:
                imgHist[m] = imgHist[m]+1
                totalExcess = totalExcess - 1 #%reduce excess
                if totalExcess == 0:
                    break
                #end
            #end
        #end

        k = k+1 #%prevent from always placing the pixels in bin #1
        if k > numBins: #% start over if numBins was reached
            k = 0
        #end
    #end
    return imgHist

#%-----------------------------------------------------------------------------
#% This function interpolates between neighboring tile mappings to produce a
#% new mapping in order to remove artificially induced tile borders.
#% Otherwise, these borders would become quite visible.  The resulting
#% mapping is applied to the input image thus producing a CLAHE processed
#% image.

def makeClaheImage(I, tileMappings, numTiles, selectedRange, numBins, dimTile):

    #%initialize the output image to zeros (preserve the class of the input image)
    claheI = np.zeros_like(I)

    #%compute the LUT for looking up original image values in the tile mappings,
    #%which we created earlier
    if I.dtype!=np.float:
        k = np.arange(selectedRange[0], selectedRange[1]+1)
        #aLut = np.zeros(k.shape[0])
        aLut = k-selectedRange[0]
        aLut = aLut/(selectedRange[1]-selectedRange[0])
    else:
        #% remap from 0..1 to 0..numBins-1
        if numBins != 1:
            binStep = 1/(numBins-1)
            start = np.ceil(selectedRange[0]/binStep)
            stop  = np.floor(selectedRange[1]/binStep)
            k = np.arange(start,stop+1)
            aLut = np.linspace(0, 1, k.shape[0]-1)
        else:
            aLut = np.zeros[0] #%in case someone specifies numBins = 1, which is just silly
        #end
    #end

    imgTileRow=1
    for k in range(numTiles[0]+1):
        if k == 0:  #%special case: top row
            imgTileNumRows = dimTile[0]/2 #%always divisible by 2 because of padding
            mapTileRows = [0, 0]
        else:
            if k == numTiles[0]: #%special case: bottom row
                imgTileNumRows = dimTile[0]/2
                mapTileRows = [numTiles[0], numTiles[0]]
            else: #%default values
                imgTileNumRows = dimTile[0]
                mapTileRows = [k-1, k] #%[upperRow lowerRow]
            #end
        #end

        #% loop over columns of the tileMappings cell array
        imgTileCol=1
        for l in range(numTiles[1]+1):
            if l == 0: #%special case: left column
                imgTileNumCols = dimTile[1]/2
                mapTileCols = [0, 0]
            else:
                if l == numTiles[1]: #% special case: right column
                    imgTileNumCols = dimTile[1]/2;
                    mapTileCols = [numTiles[1], numTiles[1]]
                else: #%default values
                    imgTileNumCols = dimTile[1]
                    mapTileCols = [l-1, l] #% right left
                #end
            #end

            #% Extract four tile mappings
            ulMapTile = tileMappings[mapTileRows[0], mapTileCols[0]]
            urMapTile = tileMappings[mapTileRows[0], mapTileCols[1]]
            blMapTile = tileMappings[mapTileRows[1], mapTileCols[0]]
            brMapTile = tileMappings[mapTileRows[1], mapTileCols[1]]

            #% Calculate the new greylevel assignments of pixels
            #% within a submatrix of the image specified by imgTileIdx. This
            #% is done by a bilinear interpolation between four different mappings
            #% in order to eliminate boundary artifacts.

            normFactor = imgTileNumRows*imgTileNumCols #%normalization factor
            imgTileIdx = np.array([np.arange(imgTileRow,imgTileRow+imgTileNumRows), np.arange(imgTileCol,imgTileCol+imgTileNumCols)])

            imgPixVals = aLut[I[imgTileIdx[0],imgTileIdx[1]]]

            #% calculate the weights used for linear interpolation between the
            #% four mappings
            rowW = np.tile(np.arange(imgTileNumRows), (1,imgTileNumCols))
            colW = np.tile(np.arange(imgTileNumCols), (imgTileNumRows,1))
            rowRevW = np.tile(np.arange(imgTileNumRows, 0, -1),(1,imgTileNumCols))
            colRevW = np.tile(imgTileNumCols, 0, -1, (imgTileNumRows,1))

            claheI[imgTileIdx[1], imgTileIdx[2]] = \
                (rowRevW * (colRevW * ulMapTile[imgPixVals] + \
                            colW    * urMapTile[imgPixVals])+ \
                rowW    * (colRevW * blMapTile[imgPixVals] + \
                            colW    * brMapTile[imgPixVals])) \
                /normFactor

            imgTileCol = imgTileCol + imgTileNumCols
        #end #%over tile cols
    imgTileRow = imgTileRow + imgTileNumRows
    #end #%over tile rows
    return claheI

#%-----------------------------------------------------------------------------

def parseInputs(varargin):

    #narginchk(1,13);

    I = varargin[0]
    #validateattributes(I, ['uint8', 'uint16', 'double', 'int16', 'single'], ...
    #            ['real', '2d', 'nonsparse', 'nonempty'], ...
    #            mfilename, 'I', 1);

    #% convert int16 to uint16
    #if isa(I,'int16')
    #    I = images.internal.builtins.int16touint16(I);
    #    int16ClassChange = true;
    #else
    int16ClassChange = False;
    #end

    if np.any(I.shape < 2):
        raise('images:adapthisteq:inputImageTooSmall')
    #end

    #%Other options
    #%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%Set the defaults
    distribution = 'uniform'
    alpha   = 0.4

    if I.dtype==np.float:
        fullRange = np.array([0, 1])
    else:
        #fullRange[0] = I[0]         #%copy class of the input image
        fullRange = np.array([-np.Inf, np.Inf], dtype=float) #%will be clipped to min and max
        #fullRange = double(fullRange);
    #end

    selectedRange   = fullRange

    #%Set the default to 256 bins regardless of the data type;
    #%the user can override this value at any time
    numBins = 256
    normClipLimit = 0.01
    numTiles = np.array([8, 8])

    #% Pre-process the inputs
    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    dimI = I.shape
    dimTile = dimI / numTiles

    #%check if tile size is reasonable
    if np.any(dimTile < 1):
        raise('images:adapthisteq:inputImageTooSmallToSplit ' + numTiles )
    #end

    #%check if the image needs to be padded; pad if necessary;
    #%padding occurs if any dimension of a single tile is an odd number
    #%and/or when image dimensions are not divisible by the selected
    #%number of tiles
    rowDiv  = dimI[0]%numTiles[0] == 0
    colDiv  = dimI[1]%numTiles[1] == 0

    if rowDiv and colDiv:
        rowEven = dimTile[0]%2 == 0
        colEven = dimTile[1]%2 == 0
    #end

    noPadRect = {}
    if  not (rowDiv and colDiv and rowEven and colEven):
        padRow = 0
        padCol = 0

        if not rowDiv:
            rowTileDim = np.floor(dimI[0]/numTiles[0]) + 1
            padRow = rowTileDim*numTiles[0] - dimI[0]
        else:
            rowTileDim = dimI[0]/numTiles[0]
        #end

        if not colDiv:
            colTileDim = np.floor(dimI[1]/numTiles[1]) + 1
            padCol = colTileDim*numTiles[1] - dimI[1]
        else:
            colTileDim = dimI[1]/numTiles[1]
        #end

        #%check if tile dimensions are even numbers
        rowEven = rowTileDim%2 == 0
        colEven = colTileDim%2 == 0

        if not rowEven:
            padRow = padRow+numTiles[0]
        #end

        if colEven:
            padCol = padCol+numTiles[1]
        #end

        padRowPre  = np.floor(padRow/2)
        padRowPost = np.ceil(padRow/2)
        padColPre  = np.floor(padCol/2)
        padColPost = np.ceil(padCol/2)

        #I = padarray(I,[padRowPre  padColPre ],'symmetric','pre');
        #I = padarray(I,[padRowPost padColPost],'symmetric','post');
        nI = np.zeros(np.array(I.shape)+np.array([padRowPre+padRowPost, padColPre+padColPost]), dtype=I.dtype)
        nI[padRowPre:padRowPost, padColPre:padColPost] = I

        #%UL corner (Row, Col), LR corner (Row, Col)
        noPadRect['ulRow'] = padRowPre
        noPadRect['ulCol'] = padColPre
        noPadRect['lrRow'] = padRowPre+dimI[0]
        noPadRect['lrCol'] = padColPre+dimI[1]
    #end

    #%redefine this variable to include the padding
    dimI = I.shape

    #%size of the single tile
    dimTile = dimI / numTiles

    #%compute actual clip limit from the normalized value entered by the user
    #%maximum value of normClipLimit=1 results in standard AHE, i.e. no clipping;
    #%the minimum value minClipLimit would uniformly distribute the image pixels
    #%across the entire histogram, which would result in the lowest possible
    #%contrast value
    numPixInTile = np.prod(dimTile)
    minClipLimit = np.ceil(numPixInTile/numBins)
    clipLimit = minClipLimit + np.round(normClipLimit*(numPixInTile-minClipLimit))

    return I, selectedRange, fullRange, numTiles, dimTile, clipLimit, numBins, noPadRect, distribution, alpha, int16ClassChange
    #%-----------------------------------------------------------------------------
