function renderScatter(heating, cooling){
  new Chart(document.getElementById("scatterChart"),{
    type:"scatter",
    data:{
      datasets:[{
        label:"Heating vs Cooling",
        data:heating.map((h,i)=>({x:h,y:cooling[i]})),
        backgroundColor:"#6366f1"
      }]
    }
  });
}

function renderBar(data){
  new Chart(document.getElementById("barChart"),{
    type:"bar",
    data:{
      labels:Object.keys(data),
      datasets:[
        {label:"Heating",data:Object.values(data),backgroundColor:"#f97316"}
      ]
    }
  });
}