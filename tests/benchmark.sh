D=examples/test_data
: > /tmp/bench.txt
for f in 50 100 200 500; do
  for x in "" "-xc" "-dl 0 -x0" "-dl 0"; do
    s=$(date +%s.%N)
    java -Xmx2500m -jar dynadupx.jar -i $D/test.$f.tre -a "_[0-9]+$" "/delete/" $x -o /tmp/fast_$f$x.tre > /tmp/fast_$f$x.log 2>&1
    e=$(date +%s.%N)
    cost=$(grep "total cost" /tmp/fast_$f$x.log | awk '{print $3}')
    chk=$(java -Xmx2500m -jar dynadupx.jar -legacy -i $D/test.$f.tre -a "_[0-9]+$" "/delete/" -st /tmp/fast_$f$x.tre 2>&1 | grep "Total number of duplications is" | awk '{print $NF}')
    echo "test.$f ${x:-default} wall=$(echo "$e - $s" | bc) cost=$cost origScoresFastTree=$chk" >> /tmp/bench.txt
  done
done
echo DONE >> /tmp/bench.txt
