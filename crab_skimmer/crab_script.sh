echo Check if TTY
if [ "`tty`" != "not a tty" ]; then
    echo "YOU SHOULD NOT RUN THIS IN INTERACTIVE, IT DELETES YOUR LOCAL FILES"
else
    echo "ENV..................................."
    env
    echo "VOMS"
    voms-proxy-info -all
    echo "CMSSW BASE, python path, pwd, ls"
    echo $CMSSW_BASE
    echo $PYTHON3PATH
    echo $PWD
    echo $(ls)
    rm -rf $CMSSW_BASE/lib/
    rm -rf $CMSSW_BASE/src/
    rm -rf $CMSSW_BASE/module/
    rm -rf $CMSSW_BASE/python/

    echo Found Proxy in: $X509_USER_PROXY
    python3 crab_script.py $1
fi
