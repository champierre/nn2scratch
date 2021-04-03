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
  all: {
    'ja': 'の全て',
    'ja-Hira': 'のすべて',
    'en': 'all'
  },
  reset: {
    'ja': 'ラベル [LABEL] をリセット',
    'ja-Hira': 'ラベル [LABEL] をリセット',
    'en': 'reset label: [LABEL]'
  },
  resetAll: {
    'ja': '全てのラベルをリセット',
    'ja-Hira': 'すべてのラベルをリセット',
    'en': 'reset all labels'
  },
  train: {
    'ja': '訓練する',
    'ja-Hira': 'くんれんする',
    'en': 'train'
  },
  getLabel: {
    'ja': '[VALUES] のラベル',
    'ja-Hira': '[VALUES] のラベル',
    'en': 'label of [VALUES]'
  },
  defaultLabel: {
    'ja': 'りんご',
    'ja-Hira': 'りんご',
    'en': 'apple'
  }
}
const AvailableLocales = ['en', 'ja', 'ja-Hira']

class Scratch3Nn2ScratchBlocks {

    constructor (runtime) {
        this.runtime = runtime;

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
                            defaultValue: Message.defaultLabel[this._locale]
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
                            defaultValue: Message.defaultLabel[this._locale]
                        }
                    }
                },
                {
                    opcode: 'resetAll',
                    blockType: BlockType.COMMAND,
                    text: Message.resetAll[this._locale]
                },
                {
                    opcode: 'reset',
                    blockType: BlockType.COMMAND,
                    text: Message.reset[this._locale],
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: Message.defaultLabel[this._locale]
                      }
                    }
                },
                {
                    opcode: 'train',
                    blockType: BlockType.COMMAND,
                    text: Message.train[this._locale]
                },
                {
                    opcode: 'getLabel',
                    text: Message.getLabel[this._locale],
                    blockType: BlockType.REPORTER,
                    arguments: {
                        VALUES: {
                            type: ArgumentType.STRING,
                            defaultValue: 0
                        }
                    }
                },
                {
                    opcode: 'debug',
                    blockType: BlockType.COMMAND,
                    text: 'debug'
                },
            ],
            menus: {
                reset_menu: {
                    items: this.getMenu()
                }
            }
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

    reset(args) {
      const array = this.nn.neuralNetworkData.data.raw;
      this.nn.neuralNetworkData.data.raw = array.filter(item => item.ys[0] !== args.LABEL);
    }

    resetAll(args) {
      this.nn.neuralNetworkData.data.raw = [];
    }

    train() {
      if (this.nn.neuralNetworkData.data.raw.length === 0) {
        alert('[Error] Data is not added yet!')
      } else {
        this.nn.normalizeData()
        const trainingOptions = {
          epochs: 32,
          batchSize: 12
        }

        this.nn.train(trainingOptions, function() {
          alert('Training is completed.');
        });
      }
    }

    getLabel(args) {
      let inputs = args.VALUES.split(',').map(v => Number(v));
      const result = this.nn.classifySync(inputs);
      let maxConfidence = 0;
      let label = null;
      for (let i = 0; i < result.length; i++) {
        if (result[i].confidence > maxConfidence) {
          maxConfidence = result[i].confidence;
          label = result[i].label;
        }
      }
      return label;
    }

    setLocale() {
      let locale = formatMessage.setup().locale;
      if (AvailableLocales.includes(locale)) {
        return locale;
      } else {
        return 'en';
      }
    }

    getMenu() {
      let arr = [];
      let defaultValue = 'all';
      let text = Message.all[this._locale];
      arr.push({text: text, value: defaultValue});
      for(let i = 1; i <= 10; i++) {
        let obj = {};
        obj.text = i.toString(10);
        obj.value = i.toString(10);
        arr.push(obj);
      };
      return arr;
    }

    debug() {
      console.log(this.nn.neuralNetworkData.data.raw);
    }
}

module.exports = Scratch3Nn2ScratchBlocks;