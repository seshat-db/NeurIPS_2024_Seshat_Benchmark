set datafile sep ','

se ytics nomirror
se y2tics 2000
se yr [0:0.64]   
se ke outside right center reverse Left spacing 1.5  

se xtics ("10,000 BCE" -10000, "8,000 BCE" -8000, "6,000 BCE" -6000, "4,000 BCE" -4000, "2,000 BCE" -2000, "0" 0, "2000 CE" 2000)
se xr [-10001:2001]

p 'ba_time.csv' u (($1+$2)/2):4 w l lw 4 lc '0xa6cee3' t 'GPT-3.5'
rep 'ba_time.csv' u (($1+$2)/2):5 w l lw 4  lc '0x1f78b4' t 'GPT-4-turbo'
rep 'ba_time.csv' u (($1+$2)/2):6 w l lw 4  lc '0xb2df8a' t 'GPT-4o'
rep 'ba_time.csv' u (($1+$2)/2):7 w l lw 4  lc '0x33a02c' t 'Llama-3-70b-chat'
rep 'ba_time.csv' u (($1+$2)/2):3 axes x1y2 w l lw 4 lc -1 t 'Number of data points'

se te post color eps solid size 5.5,2.4 "Helvetica" 12
se out 'ba_time.eps'
rep
se out
se te wxt

