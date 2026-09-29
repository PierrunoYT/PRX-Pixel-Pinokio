const assert = require('node:assert/strict')
const test = require('node:test')
const launcher = require('../pinokio')

test('partial installations cannot be launched', async () => {
  const menu = await launcher.menu(null, {
    exists: path => path === 'app/env', running: () => false,
  })
  assert.equal(menu[0].href, 'install.js')
})

for (const action of ['update', 'reset']) {
  test(`${action} remains visible while the environment is absent`, async () => {
    const menu = await launcher.menu(null, {
      exists: () => false, running: path => path === `${action}.js`,
    })
    assert.equal(menu[0].href, `${action}.js`)
  })
}

test('ready server becomes the default menu entry', async () => {
  const menu = await launcher.menu(null, {
    exists: () => true, running: path => path === 'start.js',
    local: () => ({ url: 'http://127.0.0.1:7861' }),
  })
  assert.equal(menu[0].href, 'http://127.0.0.1:7861')
  assert.equal(menu[0].default, true)
})

function evaluate(template, values) {
  return Function(...Object.keys(values), `return (${template.slice(2, -2)})`)(...Object.values(values))
}

test('one compatible torch install is selected per platform', () => {
  const cases = [
    ['win32', 'x64', 'nvidia', '/cu128'],
    ['linux', 'x64', 'nvidia', '/cu128'],
    ['linux', 'x64', 'amd', '/rocm6.3'],
    ['win32', 'x64', 'amd', '/cpu'],
    ['linux', 'x64', null, '/cpu'],
    ['darwin', 'arm64', 'apple', 'torch==2.7.0'],
    ['darwin', 'x64', null, 'Intel Macs'],
  ]
  for (const [platform, arch, gpu, expected] of cases) {
    const selected = []
    for (const step of require('../torch').run) {
      if (step.when && !evaluate(step.when, { platform, arch, gpu })) continue
      selected.push(step.params.message)
      if (step.next === null) break
    }
    assert.equal(selected.length, 1)
    assert.ok(selected[0].includes(expected))
    assert.ok(!selected[0].includes('--no-deps'))
  }
})

test('blank tokens preserve credentials; supplied tokens append a separate line', () => {
  const step = require('../install').run.find(step => step.method === 'fs.append')
  for (const hf_token of [undefined, '', '  ']) {
    assert.ok(!evaluate(step.when, { input: { hf_token } }))
  }
  const text = step.params.text.replace(/{{(.*?)}}/g, (_, code) =>
    Function('input', `return (${code})`)({ hf_token: ' hf_example\n' }))
  assert.equal(text, '\nHF_TOKEN=hf_example\n')
})

test('server URL capture excludes terminal escape codes', () => {
  const step = require('../start').run.find(step => step.method === 'shell.run')
  const regex = new RegExp(step.params.on[0].event.slice(1, -1))
  assert.equal(regex.exec('Running on local URL: http://127.0.0.1:7860\x1b[0m')[1], 'http://127.0.0.1:7860')
})

test('update refreshes diffusers without upgrading unrelated dependencies', () => {
  const commands = require('../update').run.flatMap(step => [].concat(step.params.message || []))
  const diffusers = commands.find(command => command.includes('diffusers.git'))
  assert.ok(diffusers.includes('--upgrade-package diffusers'))
  assert.ok(!/--upgrade(?!-package)|\s-U\b/.test(diffusers))
})
