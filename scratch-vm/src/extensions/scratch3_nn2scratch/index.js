const ArgumentType = require('../../extension-support/argument-type')
const BlockType = require('../../extension-support/block-type')
const Cast = require('../../util/cast')
const formatMessage = require('format-message')
const ml5 = require('ml5')

const Message = {
  normalizeData: {
    'ja': 'データを正規化する',
    'ja-Hira': 'データをせいきかする',
    'en': 'normalize data'
  },
  train: {
    'ja': '訓練する',
    'ja-Hira': 'くんれんする',
    'en': 'train'
  },
  setInputs: {
    'ja': '入力データの [KEY] に [VALUE] をセットする',
    'ja-Hira': 'にゅうりょくデータの [KEY] に [VALUE] をセットする',
    'en': 'set [VALUE] to [KEY] of inputs data'
  },
  setOutput: {
    'ja': '出力データの [KEY] に [VALUE] をセットする',
    'ja-Hira': 'しゅつりょくデータの [KEY] に [VALUE] をセットする',
    'en': 'set [VALUE] to [KEY] of output data'
  },
  addData: {
    'ja': 'データを追加する',
    'ja-Hira': 'データを追加する',
    'en': 'addData'
  },
  setClassificationInputs: {
    'ja': '分類用入力データの [KEY] に [VALUE] をセットする',
    'ja-Hira': 'ぶんるいようにゅうりょくデータの [KEY] に [VALUE] をセットする',
    'en': 'set [VALUE] to [KEY] of classification inputs data'
  },
  classify: {
    'ja': '分類する',
    'ja-Hira': 'ぶんるいする',
    'en': 'classify'
  },
  getLabel: {
    'ja': 'ラベル',
    'ja-Hira': 'ラベル',
    'en': 'label'
  },
}
const AvailableLocales = ['en', 'ja', 'ja-Hira']

class Scratch3Nn2ScratchBlocks {

    constructor (runtime) {
        this.runtime = runtime;
        this.inputs = {};
        this.output = {};
        this.classificationInputs = {};
        this.label = null;

        const options = {
          task: 'classification'
        }
        this.nn = ml5.neuralNetwork(options);
    }

    getInfo () {
        this._locale = this.setLocale();

        return {
            id: 'nn2scratch',
            name: 'Nn2Scratch',
            blocks: [
                {
                    opcode: 'normalizeData',
                    blockType: BlockType.COMMAND,
                    text: Message.normalizeData[this._locale]
                },
                {
                    opcode: 'train',
                    blockType: BlockType.COMMAND,
                    text: Message.train[this._locale]
                },
                {
                    opcode: 'setInputs',
                    blockType: BlockType.COMMAND,
                    text: Message.setInputs[this._locale],
                    arguments: {
                        KEY: {
                            type: ArgumentType.STRING
                        },
                        VALUE: {
                            type: ArgumentType.NUMBER
                        }
                    }
                },
                {
                    opcode: 'setOutput',
                    blockType: BlockType.COMMAND,
                    text: Message.setOutput[this._locale],
                    arguments: {
                        KEY: {
                            type: ArgumentType.STRING
                        },
                        VALUE: {
                            type: ArgumentType.STRING
                        }
                    }
                },
                {
                    opcode: 'addData',
                    blockType: BlockType.COMMAND,
                    text: Message.addData[this._locale]
                },
                {
                    opcode: 'setClassificationInputs',
                    blockType: BlockType.COMMAND,
                    text: Message.setClassificationInputs[this._locale],
                    arguments: {
                        KEY: {
                            type: ArgumentType.STRING
                        },
                        VALUE: {
                            type: ArgumentType.STRING
                        }
                    }
                },
                {
                    opcode: 'classify',
                    blockType: BlockType.COMMAND,
                    text: Message.classify[this._locale]
                },
                {
                    opcode: 'getLabel',
                    text: Message.getLabel[this._locale],
                    blockType: BlockType.REPORTER
                },
            ]
        };
    }

    normalizeData() {
      this.nn.normalizeData()
    }

    train() {
      const trainingOptions = {
        epochs: 32,
        batchSize: 12
      }

      this.nn.train(trainingOptions, function() {
        console.log('Training is completed.')
      });
    }

    setInputs(args) {
      this.inputs[args.KEY] = Number(args.VALUE);
    }

    setOutput(args) {
      this.output[args.KEY] = args.VALUE;
    }

    addData() {
      this.nn.addData(this.inputs, this.output);
      console.log("addData:");
      console.log("inputs", this.inputs);
      console.log("output", this.output);
      this.inputs = {};
      this.output = {};
    }

    setClassificationInputs(args) {
      console.log("setClassificationInputs:")
      console.log(`inputs[${args.KEY}]=${Number(args.VALUE)}`);
      this.classificationInputs[args.KEY] = Number(args.VALUE);
    }

    classify() {
      this.nn.classify(this.classificationInputs, (error, result) => {
        if(error){
          console.error(error);
          return;
        }
        let maxConfidence = 0;
        let label = null;
        for (let i = 0; i < result.length; i++) {
          console.log(result[i].confidence)
          if (result[i].confidence > maxConfidence) {
            maxConfidence = result[i].confidence;
            label = result[i].label;
          }
        }
        this.label = label;
        this.classificationInputs = {};
      });
    }

    getLabel() {
      return this.label;
    }

    setLocale() {
      let locale = formatMessage.setup().locale;
      if (AvailableLocales.includes(locale)) {
        return locale;
      } else {
        return 'en';
      }
    }
}

module.exports = Scratch3Nn2ScratchBlocks;
