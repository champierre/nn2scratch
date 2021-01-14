const ArgumentType = require('../../extension-support/argument-type')
const BlockType = require('../../extension-support/block-type')
const Cast = require('../../util/cast')
const formatMessage = require('format-message')
const ml5 = require('ml5')

const Message = {
  joinWithComma: {
    'ja': '[STR1] , [STR2]',
    'ja-Hira': '[STR1] , [STR2]',
    'en': '[STR1] , [STR2]'
  },
  setLabel: {
    'ja': '[VALUES] に [LABEL] とラベル付けする',
    'ja-Hira': '[VALUES] に [LABEL] とラベル付けする',
    'en': 'set label: [LABEL] for data: [VALUES]'
  },
  dataCount: {
    'ja': 'ラベル [LABEL] のデータ数',
    'ja-Hira': 'ラベル [LABEL] のデータすう',
    'en': 'data count of label: [LABEL]'
  },
  resetAllLabels: {
    'ja': 'ラベルの全てのデータをリセット',
    'ja-Hira': 'ラベルのすべてのデータをリセット',
    'en': 'reset all labels'
  },
  train: {
    'ja': '訓練する',
    'ja-Hira': 'くんれんする',
    'en': 'train'
  },
  classify: {
    'ja': '[VALUES] を分類する',
    'ja-Hira': '[VALUES] をぶんるいする',
    'en': 'classify [VALUES]'
  },
  getLabel: {
    'ja': 'ラベル',
    'ja-Hira': 'ラベル',
    'en': 'label'
  },
  label: {
    'ja': 'りんご',
    'ja-Hira': 'りんご',
    'en': 'apple'
  }
}
const AvailableLocales = ['en', 'ja', 'ja-Hira']

class Scratch3Nn2ScratchBlocks {

    constructor (runtime) {
        this.runtime = runtime;
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
                    opcode: 'joinWithComma',
                    blockType: BlockType.REPORTER,
                    text: Message.joinWithComma[this._locale],
                    arguments: {
                        STR1: {
                            type: ArgumentType.STRING,
                            defaultValue: 0
                          },
                        STR2: {
                            type: ArgumentType.STRING,
                            defaultValue: 0
                        }
                    }
                },
                {
                    opcode: 'setLabel',
                    blockType: BlockType.COMMAND,
                    text: Message.setLabel[this._locale],
                    arguments: {
                        VALUES: {
                            type: ArgumentType.STRING,
                            defaultValue: 0
                        },
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: Message.label[this._locale]
                        }
                    }
                },
                {
                    opcode: 'dataCount',
                    blockType: BlockType.REPORTER,
                    text: Message.dataCount[this._locale],
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: Message.label[this._locale]
                        }
                    }
                },
                {
                    opcode: 'resetAllLabels',
                    blockType: BlockType.COMMAND,
                    text: Message.resetAllLabels[this._locale]
                },
                {
                    opcode: 'train',
                    blockType: BlockType.COMMAND,
                    text: Message.train[this._locale]
                },
                {
                    opcode: 'classify',
                    blockType: BlockType.COMMAND,
                    text: Message.classify[this._locale],
                    arguments: {
                        VALUES: {
                            type: ArgumentType.STRING,
                            defaultValue: 0
                        }
                    }
                },
                {
                    opcode: 'getLabel',
                    text: Message.getLabel[this._locale],
                    blockType: BlockType.REPORTER
                },
            ]
        };
    }

    joinWithComma(args) {
      return `${args.STR1},${args.STR2}`;
    }

    setLabel(args) {
      let inputs = args.VALUES.split(',').map(v => Number(v));
      let outputs = [args.LABEL]

      this.nn.addData(inputs, outputs);
    }

    dataCount(args) {
      const arr = this.nn.neuralNetworkData.data.raw;
      return arr.filter(item => item.ys[0] === args.LABEL).length;
    }

    resetAllLabels() {
      this.nn.neuralNetworkData.data.raw = [];
    }

    train() {
      this.nn.normalizeData()
      const trainingOptions = {
        epochs: 32,
        batchSize: 12
      }

      this.nn.train(trainingOptions, function() {
        alert('Training is completed.');
      });
    }

    classify(args) {
      let inputs = args.VALUES.split(',').map(v => Number(v));
      this.nn.classify(inputs, (error, result) => {
        if(error){
          console.error(error);
          return;
        }

        let maxConfidence = 0;
        let label = null;
        for (let i = 0; i < result.length; i++) {
          if (result[i].confidence > maxConfidence) {
            maxConfidence = result[i].confidence;
            label = result[i].label;
          }
        }
        this.label = label;
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
