import init, * as sim from "lib-simulation-wasm";

await init() // must initalize binding before using package :) 
import { areSameAnimals } from "./helpers";





let simulation = new sim.Simulation();

document.getElementById("train").onclick = function () {
  const stat = simulation.train()
  display_stat(stat)
}



const statport = document.getElementById("statport")
function display_stat(stat) {
  const statElem = document.createElement("p")
  statElem.textContent = stat
  statport.appendChild(statElem)
}

const viewportWidth = 800;
const viewportHeight = 800;


CanvasRenderingContext2D.prototype.drawTriangle =
  function (x, y, size, rotation, color) {
    this.beginPath();

    this.moveTo(
      x - Math.sin(rotation) * size * 1.5, // constrewing triangle to get orientation
      y + Math.cos(rotation) * size * 1.5,
    );

    this.lineTo(
      x - Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
      y + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
      x - Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
      y + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
      x - Math.sin(rotation) * size * 1.5,
      y + Math.cos(rotation) * size * 1.5,
    );

    this.stroke();
    this.fillStyle = color;
    this.fill();
  };

CanvasRenderingContext2D.prototype.drawCircle =
  function (x, y, radius) {
    this.beginPath();
    this.arc(x, y, radius, 0, 2.0 * Math.PI);
    this.fillStyle = "rgb(0,0,0)";
    this.fill();

    this.fillStyle = 'rgb(0, 255, 128)';
    this.fill();
  }





function place_sims(ids) {
  // takes an array of ID's and p[laces them into div with id sim-container
  const container = document.getElementById("sim-container")
  for (const id of ids) {
    const canvas = document.createElement("canvas")
    canvas.id = id
    canvas.width = viewportWidth
    canvas.height = viewportHeight
    container.appendChild(canvas)
  }

}


function draw_world(world) {
  console.log("ID")
  console.log(world.id)
  const viewport = document.getElementById(world.id);
  const ctxt = viewport.getContext('2d');
  ctxt.fillStyle = 'rgb(0, 0, 0)';

  ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

  const apex = world.animals.reduce(
    (max, animal) => (animal.fitness > max.fitness ? animal : max),
    world.animals[0]
  );


  for (const food of world.foods) {
    ctxt.drawCircle(
      food.x * viewportWidth,
      food.y * viewportHeight,
      (0.01 / 2.0) * viewportWidth
    )
  }

  for (const animal of world.animals) {
    ctxt.drawTriangle(
      animal.x * viewportWidth,
      animal.y * viewportHeight,
      0.01 * viewportWidth,
      animal.rotation,
      areSameAnimals(apex, animal) ? 'rgb(136,8,8)' : 'rgb(255, 255, 255)'
    );

  }
}





function redraw_engine() {
  eng.step_all()

  const worlds = eng.worlds()
  const world_count = worlds.length

  worlds.forEach(w => draw_world(w))

  requestAnimationFrame(redraw_engine);
}




// just fill up with canvas. If overflows will move onto next row.
// Have a maximum of 24 simulations running.  
// Be able to call your simulations here. 
// let id of each canvas be sims id. 
// draw_world takes the id and geneates it.
function main() {

  const ids = eng.worlds().map(w => w.id)
  // get ids from worlds
  place_sims(ids)
  redraw_engine(eng)

}

main()
