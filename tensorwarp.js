/*
░██████████                                                     ░██       ░██                                
    ░██                                                         ░██       ░██                                
    ░██     ░███████  ░████████   ░███████   ░███████  ░██░████ ░██  ░██  ░██  ░██████   ░██░████ ░████████  
    ░██    ░██    ░██ ░██    ░██ ░██        ░██    ░██ ░███     ░██ ░████ ░██       ░██  ░███     ░██    ░██ 
    ░██    ░█████████ ░██    ░██  ░███████  ░██    ░██ ░██      ░██░██ ░██░██  ░███████  ░██      ░██    ░██ 
    ░██    ░██        ░██    ░██        ░██ ░██    ░██ ░██      ░████   ░████ ░██   ░██  ░██      ░███   ░██ 
    ░██     ░███████  ░██    ░██  ░███████   ░███████  ░██      ░███     ░███  ░█████░██ ░██      ░██░█████  
                                                                                                  ░██        
                                                                                                  ░██        
                                                                                                             by cerulean
*/

(async () => {
    if (!window.tf) {
        await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs';
            s.onload = resolve;
            s.onerror = reject;
            document.head.appendChild(s);
        });
    }

    if (!Scratch.extensions.unsandboxed) return console.error('TensorWarp cannot run in sandboxed mode.');

    class TensorWarp {
        constructor() {
            this.tensors = {};
            this.models = {};
            this.lastCreatedTensor = null;
        }

        getInfo() {
            return {
                id: 'tensorwarp',
                name: 'TensorWarp ML',
                color1: '#03409a',
                color2: '#0063ba',
                 blocks: [
                    {opcode:'createTensor', blockType: Scratch.BlockType.COMMAND, text:'create tensor [NAME] with values [VALUES] and shape [SHAPE]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'},
                            VALUES:{type:Scratch.ArgumentType.STRING, defaultValue:'1,2,3'},
                            SHAPE:{type:Scratch.ArgumentType.STRING, defaultValue:'3'}
                        }
                    },
                    {opcode:'viewTensorInfo', blockType: Scratch.BlockType.REPORTER, text:'view info for tensor [NAME]',
                        arguments:{NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'}}
                    },
                    {opcode:'reshapeTensor', blockType: Scratch.BlockType.COMMAND, text:'reshape tensor [NAME] to [SHAPE]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'},
                            SHAPE:{type:Scratch.ArgumentType.STRING, defaultValue:'3,1'}
                        }
                    },
                    {opcode:'disposeTensor', blockType: Scratch.BlockType.COMMAND, text:'dispose tensor [NAME]',
                        arguments:{NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'}}
                    },
                    {opcode:'createModel', blockType: Scratch.BlockType.COMMAND, text:'create model [NAME] type [TYPE]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'},
                            TYPE:{type:Scratch.ArgumentType.STRING, menu:'modelTypes', defaultValue:'SEQUENTIAL'}
                        }
                    },
                    {opcode:'addLayer', blockType: Scratch.BlockType.COMMAND, text:'add layer [LAYER_TYPE] to model [NAME] units/filters [UNITS] kernel [KERNEL] activation [ACTIVATION] input shape [INPUT_SHAPE]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'},
                            LAYER_TYPE:{type:Scratch.ArgumentType.STRING, menu:'layerTypes', defaultValue:'dense'},
                            UNITS:{type:Scratch.ArgumentType.NUMBER, defaultValue:1},
                            KERNEL:{type:Scratch.ArgumentType.STRING, defaultValue:'3,3'},
                            ACTIVATION:{type:Scratch.ArgumentType.STRING, menu:'activations', defaultValue:'relu'},
                            INPUT_SHAPE:{type:Scratch.ArgumentType.STRING, defaultValue:''}
                        }
                    },
                    {opcode:'compileModel', blockType: Scratch.BlockType.COMMAND, text:'compile model [NAME] optimizer [OPT] loss [LOSS]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'},
                            OPT:{type:Scratch.ArgumentType.STRING, menu:'optimizers', defaultValue:'sgd'},
                            LOSS:{type:Scratch.ArgumentType.STRING, menu:'losses', defaultValue:'meanSquaredError'}
                        }
                    },
                    {opcode:'trainModel', blockType: Scratch.BlockType.COMMAND, text:'train model [NAME] with X [X_TENSOR] Y [Y_TENSOR] epochs [EPOCHS]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'},
                            X_TENSOR:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'},
                            Y_TENSOR:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorY'},
                            EPOCHS:{type:Scratch.ArgumentType.NUMBER, defaultValue:10}
                        }
                    },
                    {opcode:'predictModel', blockType: Scratch.BlockType.REPORTER, text:'predict using model [NAME] input [INPUT_TENSOR]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'},
                            INPUT_TENSOR:{type:Scratch.ArgumentType.STRING, defaultValue:'tensorX'}
                        }
                    },
                    {opcode:'viewModelLayers', blockType: Scratch.BlockType.REPORTER, text:'view layers of model [NAME]',
                        arguments:{
                            NAME:{type:Scratch.ArgumentType.STRING, defaultValue:'model1'}
                        }
                    }
                ],
                menus:{
                    activations:['relu','sigmoid','tanh','softmax','linear'],
                    optimizers:['sgd','adam','rmsprop','adagrad'],
                    losses:['meanSquaredError','categoricalCrossentropy','binaryCrossentropy'],
                    modelTypes:['SEQUENTIAL','FUNCTIONAL'],
                    layerTypes:['dense','conv2d','flatten','dropout','maxPooling2d','activation']
                }
            };
        }

        _parseNumberList(str) { return String(str).split(',').map(x => Number(x.trim())); }

        createTensor(args) {
            const tensor = tf.tensor(this._parseNumberList(args.VALUES), this._parseNumberList(args.SHAPE));
            this.tensors[args.NAME] = tensor;
            this.lastCreatedTensor = args.NAME;
            return `Tensor "${args.NAME}" created`;
        }

        viewTensorInfo(args) {
            const t = this.tensors[args.NAME];
            if (!t) return 'Tensor not found';
            return `Shape: [${t.shape}], Values: ${t.arraySync()}`;
        }

        reshapeTensor(args) {
            const t = this.tensors[args.NAME];
            if (!t) return 'Tensor not found';
            this.tensors[args.NAME] = t.reshape(this._parseNumberList(args.SHAPE));
            return 'Tensor reshaped';
        }

        disposeTensor(args) {
            const t = this.tensors[args.NAME];
            if (!t) return 'Tensor not found';
            t.dispose();
            delete this.tensors[args.NAME];
            if (this.lastCreatedTensor === args.NAME) this.lastCreatedTensor = null;
            return 'Tensor disposed';
        }

        createModel(args) {
            if (args.TYPE.toUpperCase() === 'SEQUENTIAL') this.models[args.NAME] = tf.sequential();
            else if (args.TYPE.toUpperCase() === 'FUNCTIONAL') this.models[args.NAME] = {layers: [], functional:true};
            else return 'Unsupported model type';
            return 'Model created';
        }

        addLayer(args) {
            const model = this.models[args.NAME];
            if (!model) return 'Model not found';
            const type = args.LAYER_TYPE;
            let cfg = {units:Number(args.UNITS), activation:args.ACTIVATION};

            if (model.functional) { model.layers.push({type, cfg}); return 'Layer added (functional)'; }

            switch(type) {
                case 'dense':
                    if (!model.layers.length && args.INPUT_SHAPE) cfg.inputShape = this._parseNumberList(args.INPUT_SHAPE);
                    else if (!model.layers.length && this.lastCreatedTensor) cfg.inputShape = this.tensors[this.lastCreatedTensor].shape.slice(-1);
                    model.add(tf.layers.dense(cfg));
                    break;
                case 'conv2d':
                    cfg.kernelSize = this._parseNumberList(args.KERNEL);
                    cfg.filters = cfg.units;
                    if (!model.layers.length && args.INPUT_SHAPE) cfg.inputShape = this._parseNumberList(args.INPUT_SHAPE);
                    model.add(tf.layers.conv2d(cfg));
                    break;
                case 'flatten': model.add(tf.layers.flatten()); break;
                case 'dropout': model.add(tf.layers.dropout({rate:cfg.units})); break;
                case 'maxPooling2d': model.add(tf.layers.maxPooling2d({poolSize:this._parseNumberList(args.KERNEL)})); break;
                case 'activation': model.add(tf.layers.activation({activation:cfg.activation})); break;
                default: return 'Unknown layer type';
            }
            return 'Layer added';
        }

        compileModel(args) {
            const m = this.models[args.NAME];
            if (!m) return 'Model not found';
            if (!m.functional) m.compile({optimizer: args.OPT, loss: args.LOSS});
            return 'Model compiled';
        }

        async trainModel(args) {
            const m = this.models[args.NAME];
            const X = this.tensors[args.X_TENSOR], Y = this.tensors[args.Y_TENSOR];
            if (!m || !X || !Y) return 'Missing model or tensors';
            if (!m.fit) return 'Functional models cannot be trained yet';
            await m.fit(X,Y,{epochs:Number(args.EPOCHS)});
            return 'Model trained';
        }

        predictModel(args) {
            const m = this.models[args.NAME], input = this.tensors[args.INPUT_TENSOR];
            if (!m || !input) return 'Missing model or tensor';
            if (!m.predict) return 'Functional model prediction not implemented';
            return JSON.stringify(m.predict(input).arraySync());
        }

        viewModelLayers(args) {
            const m = this.models[args.NAME];
            if (!m || !m.layers.length) return 'No layers';
            return m.layers.map(l => `${l.name||l.type} (${l.units||l.filters||'?'}) input: ${JSON.stringify(l.batchInputShape?.slice(1)||l.config?.inputShape||'?')}`).join(', ');
        }
    }

    const twInstance = new TensorWarp();
    Scratch.extensions.register(twInstance);

    function openTWWindow(tw) {
        if (document.getElementById("tensorwarp-gui")) return;

        const gui = document.createElement("div");
        gui.id = "tensorwarp-gui";
        Object.assign(gui.style, {
            position:"fixed", top:"50px", left:"50px", width:"700px", height:"600px",
            background:"#0d0d0d", color:"#eee", border:"2px solid #03409A",
            borderRadius:"12px", zIndex:999999, fontFamily:"Arial,sans-serif",
            overflow:"hidden", resize:"both"
        });

        const header = document.createElement("div");
        Object.assign(header.style,{cursor:"move", padding:"12px 15px", background:"#03409A", color:"#fff", fontWeight:"bold"});
        header.innerText = "■ TensorWarp Dashboard";
        gui.appendChild(header);

        const content = document.createElement("div");
        Object.assign(content.style,{padding:"15px", height:"calc(100% - 52px)", overflow:"auto"});
        content.innerHTML = `
        <div style="display:flex; gap:12px; height:100%;">
            <div style="flex:1; overflow-y:auto;">
                <h3>Tensors</h3><div id="tw-tensors" style="background:#111;padding:6px;border-radius:6px;max-height:300px;overflow-y:auto;"></div>
                <h3>Models</h3><div id="tw-models" style="background:#111;padding:6px;border-radius:6px;max-height:250px;overflow-y:auto;"></div>
            </div>
            <div style="flex:1; overflow-y:auto;">
                <h3>Layer Viewer</h3><div id="tw-layers" style="background:#111;padding:6px;border-radius:6px;max-height:100%;overflow-y:auto;"></div>
            </div>
        </div>
        <button id="tw-close" style="margin-top:12px;padding:8px 12px;background:#7a0000;color:#fff;border:none;border-radius:6px;cursor:pointer;">Close</button>
        `;
        gui.appendChild(content);
        document.body.appendChild(gui);
        gui.querySelector("#tw-close").onclick = () => gui.remove();

        let drag=false, offX=0, offY=0;
        header.onmousedown = e => { drag=true; offX=e.clientX-gui.offsetLeft; offY=e.clientY-gui.offsetTop; };
        window.onmousemove = e => { if(drag){ gui.style.left=(e.clientX-offX)+'px'; gui.style.top=(e.clientY-offY)+'px'; } };
        window.onmouseup = () => drag=false;

        async function refreshUI() {
            const tDiv = gui.querySelector("#tw-tensors");
            tDiv.innerHTML = Object.keys(tw.tensors).length
                ? (await Promise.all(Object.keys(tw.tensors).map(async name=>{
                    const t = tw.tensors[name];
                    let preview="";
                    try {
                        const arr = await t.array();
                        const flat = arr.flat();
                        preview = `<div style="height:20px; display:flex; gap:1px;">${flat.map(v=>`<div style="flex:1; background:#03409A; height:${Math.min(20,Math.max(2,v*5))}px;"></div>`).join("")}</div>`;
                    } catch { preview="cannot preview"; }
                    return `<div style="padding:4px;border-bottom:1px solid #03409A;">
                        <b style="color:#fff;">${name}</b><br>
                        shape: <span style="color:#999;">[${t.shape}]</span><br>
                        dtype: <span style="color:#999;">${t.dtype}</span><br>
                        preview: ${preview}
                    </div>`;
                }))).join("")
                : "<i>No tensors</i>";

            const mDiv = gui.querySelector("#tw-models");
            mDiv.innerHTML = Object.keys(tw.models).length
                ? Object.keys(tw.models).map(name=>{
                    const model = tw.models[name];
                    return `<div style="padding:4px; cursor:pointer;" data-model="${name}">
                        <b style="color:#fff;">${name}</b> (${model.layers.length} layers)
                        <hr style="border-color:#03409A;opacity:0.3;">
                    </div>`;
                }).join("")
                : "<i>No models</i>";

            mDiv.querySelectorAll('[data-model]').forEach(div=>{
                div.onclick = ()=>{
                    const modelName = div.getAttribute('data-model');
                    const model = tw.models[modelName];
                    const layersHTML = model.layers.map(l=>`${l.name||l.type} (${l.units||l.filters||'?'}, input: ${JSON.stringify(l.batchInputShape?.slice(1)||l.config?.inputShape||'?')})`).join("<br>");
                    gui.querySelector("#tw-layers").innerHTML = layersHTML;
                };
            });

            setTimeout(refreshUI,500);
        }

        refreshUI();
    }

    openTWWindow(twInstance);
})();
