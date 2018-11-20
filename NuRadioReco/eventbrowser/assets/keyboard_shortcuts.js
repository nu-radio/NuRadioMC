document.onkeydown= function(key){ reactKey(key); }
function reactKey(evt) {
   if((evt.keyCode == 107) || (evt.keyCode == 190)) {
     document.getElementById('btn-next-event').click();
   }
   if((evt.keyCode == 109) || (evt.keyCode == 188)) {
     document.getElementById('btn-previous-event').click();
   }
}