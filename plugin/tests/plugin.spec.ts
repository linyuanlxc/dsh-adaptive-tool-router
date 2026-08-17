import { describe, expect, it, vi } from 'vitest'
import { apply, recommend } from '../src/index.ts'
import { resolveConfig } from '../src/config.ts'

type PreStepHandler = (
  payload: Record<string, unknown>,
  next: () => unknown,
) => unknown | Promise<unknown>

function createHarness(config: Parameters<typeof apply>[1] = {}) {
  let eventName: string | undefined
  let handler: PreStepHandler | undefined
  const on = vi.fn((event: string, callback: PreStepHandler) => {
    eventName = event
    handler = callback
  })

  apply({ on } as unknown as Parameters<typeof apply>[0], config)

  return {
    eventName,
    on,
    run: async (payload: Record<string, unknown>, next: () => unknown) => {
      if (!handler) throw new Error('plugin did not register a pre-step handler')
      return handler(payload, next)
    },
  }
}

function payloadWithTools(
  query: string,
  schemas: Array<{ name: string; description?: string }>,
  restrict = vi.fn(() => vi.fn()),
) {
  return {
    payload: {
      messages: [{ content: query }],
      turn: 1,
      step: 1,
      agent: {
        ctx: {
          tools: {
            schemas: () => schemas,
            restrict,
          },
        },
      },
    },
    restrict,
  }
}

const catalog = [
  { name: 'weather_forecast', description: 'Get a weather forecast for a city' },
  { name: 'stock_price', description: 'Get the current stock market price' },
  { name: 'tool_search', description: 'Search for more tools' },
]

describe('plugin waterfall contract', () => {
  it('registers exactly one agent/pre-step listener', () => {
    const harness = createHarness()

    expect(harness.eventName).toBe('agent/pre-step')
    expect(harness.on).toHaveBeenCalledTimes(1)
  })

  it('awaits next and does not restrict tools in shadow mode', async () => {
    const harness = createHarness({ shadow: true, verbose: false })
    const { payload, restrict } = payloadWithTools('Beijing weather tomorrow', catalog)
    const next = vi.fn(async () => ({ continue: true }))

    await expect(harness.run(payload, next)).resolves.toEqual({ continue: true })
    expect(next).toHaveBeenCalledTimes(1)
    expect(restrict).not.toHaveBeenCalled()
  })

  it('restricts to alwaysAllow plus Top-K in active mode', async () => {
    const harness = createHarness({ shadow: false, topK: 1, verbose: false })
    const { payload, restrict } = payloadWithTools('Beijing weather tomorrow', catalog)
    const next = vi.fn(async () => ({ continue: true }))

    await harness.run(payload, next)

    expect(restrict).toHaveBeenCalledTimes(1)
    expect(restrict).toHaveBeenCalledWith({
      allow: ['tool_search', 'weather_forecast'],
    })
    expect(next).toHaveBeenCalledTimes(1)
  })

  it('lifts the previous restriction before applying a new one', async () => {
    const harness = createHarness({ shadow: false, topK: 1, verbose: false })
    const lift = vi.fn()
    const restrict = vi.fn(() => lift)
    const first = payloadWithTools('Beijing weather tomorrow', catalog, restrict)
    const second = payloadWithTools('AAPL share price', catalog, restrict)

    await harness.run(first.payload, async () => undefined)
    await harness.run(second.payload, async () => undefined)

    expect(lift).toHaveBeenCalledTimes(1)
    expect(restrict).toHaveBeenLastCalledWith({
      allow: ['tool_search', 'stock_price'],
    })
  })

  it('fail-opens and still calls next when ranking throws', async () => {
    const harness = createHarness({ shadow: false, verbose: false })
    const next = vi.fn(async () => ({ continue: true }))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    await expect(harness.run({
      messages: [{ content: 'weather' }],
      agent: {
        ctx: {
          tools: {
            schemas: () => {
              throw new Error('schema lookup failed')
            },
          },
        },
      },
    }, next)).resolves.toEqual({ continue: true })

    expect(next).toHaveBeenCalledTimes(1)
    warn.mockRestore()
  })
})

describe('recommend', () => {
  it('keeps reserved tools that actually exist', () => {
    const decision = recommend(
      {
        messages: [{ content: 'Beijing weather tomorrow' }],
        agent: { ctx: { tools: { schemas: () => catalog } } },
      },
      resolveConfig({ topK: 1 }),
    )

    expect(decision.recommendations[0]?.name).toBe('weather_forecast')
    expect(decision.allow).toEqual(['tool_search', 'weather_forecast'])
  })
})
