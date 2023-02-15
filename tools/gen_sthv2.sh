cd sthv2/

filename='something-something-v2-labels.json'
fileid='1dybvYcuMkKRYu3mFytUnYX-0d8UiDMeZ'
# https://drive.google.com/file/d/1dybvYcuMkKRYu3mFytUnYX-0d8UiDMeZ/view?usp=share_link
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}

filename='something-something-v2-test.json'
fileid='1X_u0mzju6J1vdRR_QxS8LvvQspoR3Lxb'
# https://drive.google.com/file/d/1X_u0mzju6J1vdRR_QxS8LvvQspoR3Lxb/view?usp=share_link
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}

filename='something-something-v2-train.json'
fileid='1GavcyLTVR9grPYK6n-t6JfJvFZnXd84L'
# https://drive.google.com/file/d/1GavcyLTVR9grPYK6n-t6JfJvFZnXd84L/view?usp=share_link
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}

filename='something-something-v2-validation.json'
fileid='1iwNgn0h5VCql93-ngroVhyjHU_uNL728'
# https://drive.google.com/file/d/1iwNgn0h5VCql93-ngroVhyjHU_uNL728/view?usp=share_link
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}

# Download zip dataset from Google Drive
filename='20bn-something-something-v2-00'
fileid='1wgDUdRFq2H1RQgriXr_IgYxbExnjETz_'
# https://drive.google.com/file/d/1wgDUdRFq2H1RQgriXr_IgYxbExnjETz_/view?usp=share_link
# https://drive.google.com/file/d/1wgDUdRFq2H1RQgriXr_IgYxbExnjETz_/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-01'
fileid='1SKeBiT2pOybAz06OFxmi-vDcOSr4i_dg'
# https://drive.google.com/file/d/1SKeBiT2pOybAz06OFxmi-vDcOSr4i_dg/view?usp=share_link
# https://drive.google.com/file/d/1SKeBiT2pOybAz06OFxmi-vDcOSr4i_dg/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-02'
fileid='1vz53UdstkbdTJ2wsBu8kplAT4gujmnFl'
# https://drive.google.com/file/d/1vz53UdstkbdTJ2wsBu8kplAT4gujmnFl/view?usp=share_link
# https://drive.google.com/file/d/1vz53UdstkbdTJ2wsBu8kplAT4gujmnFl/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-03'
fileid='1esSKQCJh-lXJysEQ5NqcSD_3htNRwiKW'
# https://drive.google.com/file/d/1esSKQCJh-lXJysEQ5NqcSD_3htNRwiKW/view?usp=share_link
# https://drive.google.com/file/d/1esSKQCJh-lXJysEQ5NqcSD_3htNRwiKW/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-04'
fileid='1d0U3nvdLwHfkd7hITD-qZN5tXIUoG8j0'
# https://drive.google.com/file/d/1d0U3nvdLwHfkd7hITD-qZN5tXIUoG8j0/view?usp=share_link
# https://drive.google.com/file/d/1d0U3nvdLwHfkd7hITD-qZN5tXIUoG8j0/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-05'
fileid='1icKw2eGUuhnKTL75GBKrgupuxCO6WgBA'
# https://drive.google.com/file/d/1icKw2eGUuhnKTL75GBKrgupuxCO6WgBA/view?usp=share_link
# https://drive.google.com/file/d/1icKw2eGUuhnKTL75GBKrgupuxCO6WgBA/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-06'
fileid='1lL6dPlpywabW4sP9LH8CoxcdKrJ9mAAV'
# https://drive.google.com/file/d/1lL6dPlpywabW4sP9LH8CoxcdKrJ9mAAV/view?usp=share_link
# https://drive.google.com/file/d/1lL6dPlpywabW4sP9LH8CoxcdKrJ9mAAV/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-07'
fileid='1_nu6iFU9OzyT-9WZP8wZn53gZgUbG3BW'
# https://drive.google.com/file/d/1_nu6iFU9OzyT-9WZP8wZn53gZgUbG3BW/view?usp=share_link
# https://drive.google.com/file/d/1_nu6iFU9OzyT-9WZP8wZn53gZgUbG3BW/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-08'
fileid='1KORSB7ncR9KvG9e8U-lbF7p0OeM_OnLN'
# https://drive.google.com/file/d/1KORSB7ncR9KvG9e8U-lbF7p0OeM_OnLN/view?usp=share_link
# https://drive.google.com/file/d/1KORSB7ncR9KvG9e8U-lbF7p0OeM_OnLN/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-09'
fileid='1-AX-2UpAw74hT1Bo9KGK1IQNVemBwgUM'
# https://drive.google.com/file/d/1-AX-2UpAw74hT1Bo9KGK1IQNVemBwgUM/view?usp=share_link
# https://drive.google.com/file/d/1-AX-2UpAw74hT1Bo9KGK1IQNVemBwgUM/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-10'
fileid='1eK341p44Uh85NUUI75di5zFKoB5CuGh5'
# https://drive.google.com/file/d/1eK341p44Uh85NUUI75di5zFKoB5CuGh5/view?usp=share_link
# https://drive.google.com/file/d/1eK341p44Uh85NUUI75di5zFKoB5CuGh5/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-11'
fileid='1UNaxnXbFeK3-k3uqMspgJpAnfU4W-Z9G'
# https://drive.google.com/file/d/1UNaxnXbFeK3-k3uqMspgJpAnfU4W-Z9G/view?usp=share_link
# https://drive.google.com/file/d/1UNaxnXbFeK3-k3uqMspgJpAnfU4W-Z9G/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-12'
fileid='1sJLnE6yCCRf0iddNXhH59vHLfn8K07Yz'
# https://drive.google.com/file/d/1sJLnE6yCCRf0iddNXhH59vHLfn8K07Yz/view?usp=share_link
# https://drive.google.com/file/d/1sJLnE6yCCRf0iddNXhH59vHLfn8K07Yz/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-13'
fileid='1ATO6Tqi-rYkZijJ3BG79yoHeOL7Adbow'
# https://drive.google.com/file/d/1ATO6Tqi-rYkZijJ3BG79yoHeOL7Adbow/view?usp=share_link
# https://drive.google.com/file/d/1ATO6Tqi-rYkZijJ3BG79yoHeOL7Adbow/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-14'
fileid='1kLBztXGBDnJ6X0f-kAUY1KqjELIO0mj1'
# https://drive.google.com/file/d/1kLBztXGBDnJ6X0f-kAUY1KqjELIO0mj1/view?usp=share_link/
# https://drive.google.com/file/d/1kLBztXGBDnJ6X0f-kAUY1KqjELIO0mj1/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-15'
fileid='1TaSiVkwku1m3l7h5lDsg8PU9bLOHILAc'
# https://drive.google.com/file/d/1TaSiVkwku1m3l7h5lDsg8PU9bLOHILAc/view?usp=share_link
# https://drive.google.com/file/d/1TaSiVkwku1m3l7h5lDsg8PU9bLOHILAc/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-16'
fileid='1mf-j9yghrOC57xqR4nmn-cJctTV5W_0W'
# https://drive.google.com/file/d/1mf-j9yghrOC57xqR4nmn-cJctTV5W_0W/view?usp=share_link
# https://drive.google.com/file/d/1mf-j9yghrOC57xqR4nmn-cJctTV5W_0W/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-17'
fileid='1P81nGTdT6P8t1W9HtkdA4blpgJlTt1mL'
# https://drive.google.com/file/d/1P81nGTdT6P8t1W9HtkdA4blpgJlTt1mL/view?usp=share_link
# https://drive.google.com/file/d/1P81nGTdT6P8t1W9HtkdA4blpgJlTt1mL/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-18'
fileid='1BC3EH8CGYwmXmlS2y2WFyMOif7lyzcAc'
# https://drive.google.com/file/d/1BC3EH8CGYwmXmlS2y2WFyMOif7lyzcAc/view?usp=share_link
# https://drive.google.com/file/d/1BC3EH8CGYwmXmlS2y2WFyMOif7lyzcAc/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

filename='20bn-something-something-v2-19'
fileid='1e3nObc-3m8BzqU5Bs3mMG4ybFohvfSrW'
# https://drive.google.com/file/d/1e3nObc-3m8BzqU5Bs3mMG4ybFohvfSrW/view?usp=share_link
# https://drive.google.com/file/d/1e3nObc-3m8BzqU5Bs3mMG4ybFohvfSrW/view?usp=share_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
