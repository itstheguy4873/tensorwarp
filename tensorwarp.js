/*
                                                                                                                                                                                                            
                                                                                                                                                                                                            
TTTTTTTTTTTTTTTTTTTTTTT                                                                                    WWWWWWWW                           WWWWWWWW                                                      
T:::::::::::::::::::::T                                                                                    W::::::W                           W::::::W                                                      
T:::::::::::::::::::::T                                                                                    W::::::W                           W::::::W                                                      
T:::::TT:::::::TT:::::T                                                                                    W::::::W                           W::::::W                                                      
TTTTTT  T:::::T  TTTTTTeeeeeeeeeeee    nnnn  nnnnnnnn        ssssssssss      ooooooooooo   rrrrr   rrrrrrrrrW:::::W           WWWWW           W:::::Waaaaaaaaaaaaa  rrrrr   rrrrrrrrr   ppppp   ppppppppp   
        T:::::T      ee::::::::::::ee  n:::nn::::::::nn    ss::::::::::s   oo:::::::::::oo r::::rrr:::::::::rW:::::W         W:::::W         W:::::W a::::::::::::a r::::rrr:::::::::r  p::::ppp:::::::::p  
        T:::::T     e::::::eeeee:::::een::::::::::::::nn ss:::::::::::::s o:::::::::::::::or:::::::::::::::::rW:::::W       W:::::::W       W:::::W  aaaaaaaaa:::::ar:::::::::::::::::r p:::::::::::::::::p 
        T:::::T    e::::::e     e:::::enn:::::::::::::::ns::::::ssss:::::so:::::ooooo:::::orr::::::rrrrr::::::rW:::::W     W:::::::::W     W:::::W            a::::arr::::::rrrrr::::::rpp::::::ppppp::::::p
        T:::::T    e:::::::eeeee::::::e  n:::::nnnn:::::n s:::::s  ssssss o::::o     o::::o r:::::r     r:::::r W:::::W   W:::::W:::::W   W:::::W      aaaaaaa:::::a r:::::r     r:::::r p:::::p     p:::::p
        T:::::T    e:::::::::::::::::e   n::::n    n::::n   s::::::s      o::::o     o::::o r:::::r     rrrrrrr  W:::::W W:::::W W:::::W W:::::W     aa::::::::::::a r:::::r     rrrrrrr p:::::p     p:::::p
        T:::::T    e::::::eeeeeeeeeee    n::::n    n::::n      s::::::s   o::::o     o::::o r:::::r               W:::::W:::::W   W:::::W:::::W     a::::aaaa::::::a r:::::r             p:::::p     p:::::p
        T:::::T    e:::::::e             n::::n    n::::nssssss   s:::::s o::::o     o::::o r:::::r                W:::::::::W     W:::::::::W     a::::a    a:::::a r:::::r             p:::::p    p::::::p
      TT:::::::TT  e::::::::e            n::::n    n::::ns:::::ssss::::::so:::::ooooo:::::o r:::::r                 W:::::::W       W:::::::W      a::::a    a:::::a r:::::r             p:::::ppppp:::::::p
      T:::::::::T   e::::::::eeeeeeee    n::::n    n::::ns::::::::::::::s o:::::::::::::::o r:::::r                  W:::::W         W:::::W       a:::::aaaa::::::a r:::::r             p::::::::::::::::p 
      T:::::::::T    ee:::::::::::::e    n::::n    n::::n s:::::::::::ss   oo:::::::::::oo  r:::::r                   W:::W           W:::W         a::::::::::aa:::ar:::::r             p::::::::::::::pp  
      TTTTTTTTTTT      eeeeeeeeeeeeee    nnnnnn    nnnnnn  sssssssssss       ooooooooooo    rrrrrrr                    WWW             WWW           aaaaaaaaaa  aaaarrrrrrr             p::::::pppppppp    
                                                                                                                                                                                         p:::::p            
                                                                                                                                                                                         p:::::p            
                                                                                                                                                                                        p:::::::p           
                                                                                                                                                                                        p:::::::p           
                                                                                                                                                                                        p:::::::p           
                                                                                                                                                                                        ppppppppp           
                                                                                                                                                                                                            by cerulean
*/

let script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs';

script.onload = () => {
    console.log('TensorFlow.js loaded');
    window.tf = tf;
};

document.head.appendChild(script);

if (!Scratch.extensions.unsandboxed) {
    throw new Error('TensorWarp does not support sandboxed mode.');
}

class tensorwarp {
    constructor() {
        this.tensors = {};
        this.models = {};
    }

    getInfo() {
        return {
            id: 'tensorwarp',
            name: 'TensorWarp ML',
            blocks: [
                {
                    opcode: 'createTensor',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'create tensor [NAME] with values [VALUES] and shape [SHAPE]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' },
                        VALUES: { type: Scratch.ArgumentType.STRING, defaultValue: '1,2,3' },
                        SHAPE: { type: Scratch.ArgumentType.STRING, defaultValue: '3' }
                    }
                },
                {
                    opcode: 'viewTensorInfo',
                    blockType: Scratch.BlockType.REPORTER,
                    text: 'view info for tensor [NAME]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' }
                    }
                },
                {
                    opcode: 'reshapeTensor',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'reshape tensor [NAME] to [SHAPE]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' },
                        SHAPE: { type: Scratch.ArgumentType.STRING, defaultValue: '3,1' }
                    }
                },
                {
                    opcode: 'disposeTensor',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'dispose tensor [NAME]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' }
                    }
                },
                {
                    opcode: 'createModel',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'create model [NAME] type [TYPE]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' },
                        TYPE: { type: Scratch.ArgumentType.STRING, menu: 'modelTypes', defaultValue: 'SEQUENTIAL' }
                    }
                },
                {
                    opcode: 'addLayer',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'add dense layer to model [NAME] units [UNITS] activation [ACTIVATION]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' },
                        UNITS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
                        ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'activations', defaultValue: 'relu' }
                    }
                },
                {
                    opcode: 'compileModel',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'compile model [NAME] optimizer [OPT] loss [LOSS]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' },
                        OPT: { type: Scratch.ArgumentType.STRING, menu: 'optimizers', defaultValue: 'sgd' },
                        LOSS: { type: Scratch.ArgumentType.STRING, menu: 'losses', defaultValue: 'meanSquaredError' }
                    }
                },
                {
                    opcode: 'trainModel',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'train model [NAME] with X [X_TENSOR] Y [Y_TENSOR] epochs [EPOCHS]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' },
                        X_TENSOR: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' },
                        Y_TENSOR: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorY' },
                        EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 10 }
                    }
                },
                {
                    opcode: 'predictModel',
                    blockType: Scratch.BlockType.REPORTER,
                    text: 'predict using model [NAME] input [INPUT_TENSOR]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' },
                        INPUT_TENSOR: { type: Scratch.ArgumentType.STRING, defaultValue: 'tensorX' }
                    }
                },
                {
                    opcode: 'viewModelLayers',
                    blockType: Scratch.BlockType.REPORTER,
                    text: 'view layers of model [NAME]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'model1' }
                    }
                }

            ],
            menus: {
                activations: ['relu', 'sigmoid', 'tanh', 'softmax', 'linear'],
                optimizers: ['sgd', 'adam', 'rmsprop', 'adagrad'],
                losses: ['meanSquaredError', 'categoricalCrossentropy', 'binaryCrossentropy'],
                modelTypes: ['SEQUENTIAL']
            }
        };
    }

    async createTensor(args) {
        try {
            const values = args.VALUES.split(',').map(v => {
                const n = Number(v.trim());
                if (isNaN(n)) throw new Error(`Invalid number: "${v}"`);
                return n;
            });
            const shape = args.SHAPE.split(',').map(n => {
                const s = Number(n.trim());
                if (isNaN(s)) throw new Error(`Invalid shape: "${n}"`);
                return s;
            });
            const tensor = tf.tensor(values, shape);
            this.tensors[args.NAME] = tensor;
            return `Tensor "${args.NAME}" created`;
        } catch (e) {
            return `Error creating tensor: ${e.message}`;
        }
    }

    async viewTensorInfo(args) {
        const tensor = this.tensors[args.NAME];
        if (!tensor) return `Tensor "${args.NAME}" not found`;
        return `Shape: [${tensor.shape}], DType: ${tensor.dtype}, Values: ${JSON.stringify(tensor.arraySync())}`;
    }

    async reshapeTensor(args) {
        const tensor = this.tensors[args.NAME];
        if (!tensor) return `Tensor "${args.NAME}" not found`;
        try {
            const newShape = args.SHAPE.split(',').map(n => Number(n.trim()));
            this.tensors[args.NAME] = tensor.reshape(newShape);
            return `Tensor "${args.NAME}" reshaped to [${newShape}]`;
        } catch (e) {
            return `Error reshaping tensor: ${e.message}`;
        }
    }

    async disposeTensor(args) {
        const tensor = this.tensors[args.NAME];
        if (!tensor) return `Tensor "${args.NAME}" not found`;
        tensor.dispose();
        delete this.tensors[args.NAME];
        return `Tensor "${args.NAME}" disposed`;
    }

    async createModel(args) {
        if (args.TYPE.toUpperCase() === 'SEQUENTIAL') {
            this.models[args.NAME] = tf.sequential();
        } else {
            return `Only SEQUENTIAL models supported`;
        }
        return `Model "${args.NAME}" created`;
    }

    async addLayer(args) {
        const model = this.models[args.NAME];
        if (!model) return `Model "${args.NAME}" not found`;
        if (!args.UNITS || args.UNITS <= 0) return `Units must be > 0`;
        if (!args.ACTIVATION) return `Activation must be specified`;

        model.add(tf.layers.dense({ units: args.UNITS, activation: args.ACTIVATION }));
        return `Added dense layer to "${args.NAME}"`;
    }

    async compileModel(args) {
        const model = this.models[args.NAME];
        if (!model) return `Model "${args.NAME}" not found`;
        try {
            model.compile({ optimizer: args.OPT, loss: args.LOSS });
            return `Model "${args.NAME}" compiled`;
        } catch (e) {
            return `Error compiling model: ${e.message}`;
        }
    }

    async trainModel(args) {
        const model = this.models[args.NAME];
        const X = this.tensors[args.X_TENSOR];
        const Y = this.tensors[args.Y_TENSOR];
        if (!model) return `Model "${args.NAME}" not found`;
        if (!X) return `Tensor "${args.X_TENSOR}" not found`;
        if (!Y) return `Tensor "${args.Y_TENSOR}" not found`;

        try {
            await model.fit(X, Y, { epochs: args.EPOCHS });
            return `Model "${args.NAME}" trained for ${args.EPOCHS} epochs`;
        } catch (e) {
            return `Error training model: ${e.message}`;
        }
    }

    async predictModel(args) {
        const model = this.models[args.NAME];
        const input = this.tensors[args.INPUT_TENSOR];
        if (!model) return `Model "${args.NAME}" not found`;
        if (!input) return `Tensor "${args.INPUT_TENSOR}" not found`;

        try {
            const output = model.predict(input);
            let result;
            if (Array.isArray(output)) {
                result = output.map(t => t.arraySync());
            } else if (output.arraySync) {
                result = output.arraySync();
            } else {
                return `Unexpected output type from model`;
            }
            return JSON.stringify(result);
        } catch (e) {
            return `Error predicting: ${e.message}`;
        }
    }

    async viewModelLayers(args) {
        const model = this.models[args.NAME];
        if (!model) return `Model "${args.NAME}" not found`;
        if (!model.layers || model.layers.length === 0) return `Model "${args.NAME}" has no layers`;

        return model.layers.map(l => {
            const units = l.units !== undefined ? l.units : (l.filters !== undefined ? l.filters : '?');
            return `${l.name} (${units} units)`;
        }).join(', ');
    }
}

Scratch.extensions.register(new tensorwarp());