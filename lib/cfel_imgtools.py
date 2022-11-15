# -*- coding: utf-8 -*-
#
#	CFEL image handling tools
#	Anton Barty
#

import numpy
import pyqtgraph.exporters



def histogram_clip_levels(data, value):
    """
    Return the top and bottom <value> percentile of data points
    Useful for scaling images to reject outliers
    Input data unchanged - leave image modification to another routine
    Return only the suggested limits 
    <value> is typically 0.001 or 0.01 (equivalent to clipping top and bottom 0.1% and 1% of pixels respectively)    
    """
    
    # Histogram bounds
    d_max = numpy.int64(numpy.nanmax(data.ravel()))
    d_min = numpy.int64(numpy.nanmin(data.ravel()))
    #print('Unclipped range = ',d_min, d_max)





    #Try histogram levels.
    # Return min/max on any error
    try:
        h_max = max(1, d_max) + 1
        h_min = min(0,d_min)
        h_nbins = max(100, h_max-h_min)
        h_nbins = min(1000000, h_nbins)
        h_range = (h_min, h_max)
        #print('nbins=', h_nbins, 'range=',h_range)


        # Cumulative histogram
        h, e = numpy.histogram(data, bins=h_nbins, range=h_range)
        c = h.cumsum()/h.sum()

        # Top and bottom <value> percentile
        w_top = numpy.where(c < (1-value))
        w_bottom = numpy.where(c > value)

        try:
            i_top = numpy.amax(w_top)
        except:
            i_top = h.size-1

        try:
            i_bottom = numpy.amin(w_bottom)
        except:
            i_bottom = 0


        bottom = e[i_bottom]
        top = max(e[i_top], bottom + 1)

    except:
        bottom = d_min
        top = d_max




    #print('Clipped range = ',bottom, top)
    #print("histogram_clip: ", bottom, top)

    # Return suggested levels for image scaling
    return bottom, top
#end histogram_clip_levels



def histogram_clip(data, value):
    """
    Apply histogram clipping to data 
    Return modified array with values clipped - use histogram_clip_levels(...) to find out the clipping levels only
    <value> is typically 0.001 or 0.01 (equivalent to clipping top and bottom 0.1% and 1% of pixels respectively)    
    """
    
    bottom, top = histogram_clip_levels(data, value)
    result = data.clip(bottom, top)
    return result

#    top, bottom = histogram_clip_levels(data, value)
#    result = data
#    result[result > top] = top
#    result[result < bottom] = bottom
#    return result
#end histogram_clip
    



def pixel_remap(data, gx, gy, dx=1.0):
    """
    :param data:
    :param gx:
    :param gy:
    :param dx:
    :return:
    """
    
    # So we don't overwrite gx, gy by accident     
    temp_x = gx / dx
    temp_y = gy / dx
     
    # Figure out array bounds
    max_x = numpy.fabs(gx).max()     
    max_y = numpy.fabs(gy).max()
    nx = int(2*max_x + 2)
    ny = int(2*max_y + 2)

    # Put (0,0) pixel at the center
    temp_x += nx//2
    temp_y += ny//2
     
    # Create array of appropriate size
    image = numpy.zeros(numpy.array([nx,ny]), dtype=numpy.float32)
     
    # IndexError: arrays used as indices must be of integer (or boolean) type
    temp_x = numpy.int_(temp_x)          
    temp_y = numpy.int_(temp_y)     
    
    # Remap to new pixels
    image[temp_x.ravel(), temp_y.ravel()] = data.ravel()
    
    # Return remapped image
    return image
#end pixel_remap



def radial_average(data, r):
    """
    In IDL
	;;
	;;	Create radial distance array
	;;
		if keyword_set(x) AND keyword_set(y) then begin
			d = sqrt(x*x+y*y)
		endif $
		else if n_elements(r) ne 0 then begin
			d = r
		endif $
		else begin
			s = size(data,/dim)
			d = dist(s[0],s[1])
			d = shift(d,s[0]/2,s[1]/2)
		endelse
  
		if not keyword_set(binsize) then $
			binsize = 1
		
	;;
	;; Ignore regions of missing data
	;;
		tempdata = data
		if keyword_set(missing) then begin
			w = where(data ne missing)
			if w[0] eq -1 then begin
				print,'radial_average: All data is missing data!'
				return, fltarr(max(d))
			endif $
			else begin
				tempdata = data[w]
				d = d[w]
			endelse
		endif

	;;
	;;	Sort distances
	;;
		dmin = 0
		dmax = max(d)
		hd = histogram(d, min=0, max=dmax, binsize=binsize, reverse_indices=ii, _extra=extra, locations=loc)

	;;
	;;	Output array
	;;
		avg = hd*0.

	;;
	;;	Compute radial average
	;;
		for i = 1, n_elements(hd)-1 do begin
		  	if ii[i] NE ii[i+1] then $
				avg[i] = total(tempdata[ii[ii[i]:ii[i+1]-1]])/hd[i]
		endfor

	;;
	;;	Return radial average
	;;
		return, avg
  """
    print("Not yet implemented...")          
#end radial_average
    

def write_png(filename, image):
    """
    ;;
    ;;	Write <image> as a .png file using the current colour table 
    ;;
    ;;	There are two approaches:
    ;;		True colour image (3-bit per pixel)
    ;;		img=tvrd(true=1) 	produces [3,nx,ny] true-colour 
    ;;		write_png, filename, img
    ;;	OR
    ;;		256-level indexed colour image
    ;;		write_png, filename, data(byte), r,g,b
    ;;
    
        
    PRO write_png, filename, img, flip=flip, rv=rv, uint16=uint16
    
    	temp = float(img)
    	if KEYWORD_SET(rv) then $
    		temp = -temp
    	if KEYWORD_SET(flip) then $
    		temp=reverse(temp,2)
    
    	;; 16-bit greyscale
    	if keyword_set(uint16) then begin
    		temp -= min(temp)
    		temp *= 65535./max(temp)
    		temp = uint(temp)
    		write_png,filename,temp
    	endif $
    	
    	;;	Colour table PNG
    	else begin
    		tvlct,r,g,b,/get
    		cts=size(r,/dim)
    		temp = bytscl(temp, top=cts[0]-1)	
    		write_png,filename,temp,r,g,b	
    	endelse
    end
    """
    
    print("Not yet implemented...")
    print(image.shape)
    print(filename)

    exporter = pyqtgraph.exporters.ImageExporter()

#end write_png

      
