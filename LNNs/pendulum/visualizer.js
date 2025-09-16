document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('pendulumCanvas');
    const ctx = canvas.getContext('2d');
    const svg = document.getElementById('networkSvg');
    const frameCounter = document.getElementById('frameCounter');

    const SVG_NS = "http://www.w3.org/2000/svg";
    let neuronNodes = {};

    fetch('visualization_data.json')
        .then(response => response.json())
        .then(data => {
            console.log("Visualization data loaded.", data);
            drawNetwork(data.network_layout);
            animate(data);
        })
        .catch(error => {
            console.error("Failed to load visualization data:", error);
            ctx.font = "16px Arial";
            ctx.fillStyle = "red";
            ctx.fillText("Error: Could not load visualization_data.json", 10, 50);
        });

    function drawNetwork(layout) {
        const PADDING = 20;
        const width = svg.getAttribute('width') - 2 * PADDING;
        const height = svg.getAttribute('height') - 2 * PADDING;

        const xCoords = Object.values(layout).map(pos => pos[0]);
        const yCoords = Object.values(layout).map(pos => pos[1]);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);
        
        const scaleX = (maxX - minX) > 0 ? width / (maxX - minX) : width;
        const scaleY = (maxY - minY) > 0 ? height / (maxY - minY) : height;

        for (const [nodeId, pos] of Object.entries(layout)) {
            const circle = document.createElementNS(SVG_NS, 'circle');
            const cx = (pos[0] - minX) * scaleX + PADDING;
            const cy = height - (pos[1] - minY) * scaleY + PADDING;
            
            circle.setAttribute('cx', cx);
            circle.setAttribute('cy', cy);
            
            if (nodeId.startsWith('in')) {
                circle.setAttribute('r', 6);
                circle.setAttribute('fill', 'skyblue');
            } else if (nodeId.startsWith('out')) {
                circle.setAttribute('r', 6);
                circle.setAttribute('fill', 'lightcoral');
            } else {
                circle.setAttribute('r', 4);
                circle.setAttribute('fill', 'gray');
            }
            
            circle.setAttribute('stroke', 'black');
            circle.setAttribute('stroke-width', 0.5);
            svg.appendChild(circle);
            neuronNodes[nodeId] = circle;
        }
    }
    
    function drawPendulum(frame, config) {
        const { cart_x, pole_theta } = frame;
        const { pole_length } = config;
        const width = canvas.width;
        const height = canvas.height;
        const scale = (height / 3) / pole_length;
        const cartWidth = 50;
        const cartHeight = 25;
        const cartY = height * 0.75;

        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, cartY);
        ctx.lineTo(width, cartY);
        ctx.stroke();

        const cartX = width / 2 + cart_x * scale;
        const poleEndX = cartX + scale * pole_length * Math.sin(pole_theta);
        const poleEndY = cartY - scale * pole_length * Math.cos(pole_theta);

        ctx.fillStyle = 'royalblue';
        ctx.fillRect(cartX - cartWidth / 2, cartY - cartHeight / 2, cartWidth, cartHeight);
        ctx.strokeStyle = 'brown';
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(cartX, cartY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();
    }
    
    const toGray = (value) => {
        const grayValue = Math.floor(value * 255);
        return `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
    };

    function animate(data) {
        let frameIndex = 0;
        const frames = data.frames;
        const totalFrames = frames.length;

        function animationLoop() {
            if (frameIndex >= totalFrames) frameIndex = 0;

            const frame = frames[frameIndex];
            drawPendulum(frame, data.config);

            const { activations, inputs, output } = frame;

            // Color hidden neurons
            for (let i = 0; i < activations.length; i++) {
                const nodeId = `h_${i}`;
                if (neuronNodes[nodeId]) neuronNodes[nodeId].setAttribute('fill', toGray(activations[i]));
            }

            // Color input neurons
            for (let i = 0; i < inputs.length; i++) {
                const nodeId = `in_${i}`;
                if (neuronNodes[nodeId]) neuronNodes[nodeId].setAttribute('fill', toGray(inputs[i]));
            }

            // Color output neurons
            const outNodeId = `out_0`;
            if (neuronNodes[outNodeId]) neuronNodes[outNodeId].setAttribute('fill', toGray(output));

            frameCounter.textContent = frameIndex;
            frameIndex++;
            requestAnimationFrame(animationLoop);
        }

        animationLoop();
    }
});