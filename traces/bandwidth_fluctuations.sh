#!/usr/bin/env bash
IFACE="wlp82s0" 

while true
do

    sudo tc qdisc del dev $IFACE root 
    sudo tc qdisc add dev $IFACE root handle 1: htb default 1 r2q 10
    sudo tc class add dev $IFACE parent 1: classid 1:1 htb rate 20mbit 
    sudo tc filter add dev $IFACE protocol ip parent 1:0 prio 1 u32 match ip dst 0.0.0.0/0 flowid 1:1 
    
    i=1 
    while IFS=, read -r tput 
    do 
        sudo tc class change dev $IFACE parent 1: classid 1:1 htb rate "$tput"kbit 
        sleep 1 
        echo "$tput" 
        
        if [ "$i" -eq 120 ]; then 
            break 
        fi 
        i=$((i+1)) 
    done < lte_cascading.csv 

    sleep 5
done