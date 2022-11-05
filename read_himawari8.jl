using CSV, DataFrames
using Plots

#correct green band
F = 0.07 #from Miller et al 2016
Gh = ( 1 - F ) .* B2 .+ F .* B4

BT = 290
if BT <= 230 P_fs = 0.3
elseif BT > 230 && BT < 280 P_fs = 0.3 + ( BT - 230 ) * (0.7 / 50 )
elseif BT >= 280 P_fs = 1.0
end

#enhancment
gamma = from_image
bandrescaled = ( ( bandoriginal / 255.0 ) ^ (1.0/gamma) ) * 255.0


band = "01"
date = 20180404
time = "0030"
foo  = "tc/data/raw/HS_H08_$(date)_$(time)_B$(band)_FLDK_R20_S0101.DAT"
out = read( foo )

y = Array{Float16}(undef, 3000,3000) 
read!("tc/data/raw/201507070200.vis.03.rfc.fld.4km.bin", y)
heatmap(y, clim = (-10,10) )




y = Array{Float16}(undef, 6000,6000) 
read!("tc/data/raw/201507070200.vis.01.fld.geoss", y)
heatmap(y, clim = (-10,10) )


ntoh.(xx)