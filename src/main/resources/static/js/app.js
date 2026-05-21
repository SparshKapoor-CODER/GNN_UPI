const themeBtn = document.getElementById("themeBtn");

themeBtn.addEventListener("click",()=>{

document.body.classList.toggle("light");

if(document.body.classList.contains("light")){

document.body.style.background="#f4f7fb";
document.body.style.color="#111827";

}else{

document.body.style.background="#081120";
document.body.style.color="white";

}

});

function log(msg){

const el=document.getElementById("log");

el.innerHTML=
'['+new Date().toLocaleTimeString()+'] '+msg+'<br>'+el.innerHTML;

}

async function refresh(){

try{

const accs=await fetch('/api/accounts').then(r=>r.json());

document.querySelector('#accounts-table tbody').innerHTML=
accs.map(a=>`

<tr>

<td>${a.vpa}</td>
<td>${a.holderName}</td>
<td>₹${parseFloat(a.balance).toFixed(2)}</td>

</tr>

`).join('');

const txs=await fetch('/api/transactions').then(r=>r.json());

document.querySelector('#tx-table tbody').innerHTML=
txs.map(t=>`

<tr>

<td>${t.id}</td>
<td>${t.senderVpa}</td>
<td>${t.receiverVpa}</td>
<td>₹${parseFloat(t.amount).toFixed(2)}</td>
<td>${t.status}</td>

</tr>

`).join('');

document.getElementById("txCount").innerText=txs.length;

document.getElementById("fraudBlocked").innerText=
txs.filter(t=>t.status==="BLOCKED_BY_GNN").length;

const mesh=await fetch('/api/mesh/state').then(r=>r.json());

document.getElementById("meshDevices").innerText=
mesh.devices.length;

}catch(e){

console.log(e);

}

}

async function sendPacket(){

const body={

senderVpa:document.getElementById('senderVpa').value,
receiverVpa:document.getElementById('receiverVpa').value,
amount:parseFloat(document.getElementById('amount').value),
pin:document.getElementById('pin').value,
ttl:5,
startDevice:'phone-alice'

};

const r=await fetch('/api/demo/send',{

method:'POST',

headers:{
'Content-Type':'application/json'
},

body:JSON.stringify(body)

}).then(r=>r.json());

log(`📤 Packet ${r.packetId.substring(0,8)} injected`);

refresh();

}

async function gossip(){

await fetch('/api/mesh/gossip',{

method:'POST'

});

log("🔄 Gossip round completed");

refresh();

}

async function flushBridges(){

const r=await fetch('/api/mesh/flush',{

method:'POST'

}).then(r=>r.json());

log(`📡 Uploaded ${r.uploadsAttempted} packets`);

refresh();

}

refresh();

setInterval(refresh,4000);