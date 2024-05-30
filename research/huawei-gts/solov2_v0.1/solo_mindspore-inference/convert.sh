export LITE_ROOT=/root/mindspore-lite-2.3.0rc2-linux-aarch64
export LD_LIBRARY_PATH=$LITE_ROOT/tools/converter/lib:$LD_LIBRARY_PATH

# dump ge graph
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=2


$LITE_ROOT/tools/converter/converter/converter_lite --fmk=MINDIR --saveType=MINDIR --optimize=ascend_oriented --modelFile=out.mindir --outputFile=out_lite --inputShape="img:1,3,800,1216"