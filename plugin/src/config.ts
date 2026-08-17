export interface RouterConfig {
  /** Observe and log recommendations without hiding tools. Default: true. */
  shadow?: boolean
  /** Number of ranked tools to recommend or expose. Default: 8. */
  topK?: number
  /** Append-only JSONL path for shadow / restrict decisions. */
  logPath?: string
  /** Tools that stay visible even when restriction is on. */
  alwaysAllow?: string[]
  /** Print one decision line per agent step. Default: true. */
  verbose?: boolean
  k1?: number
  b?: number
}

export interface ResolvedConfig {
  shadow: boolean
  topK: number
  logPath?: string
  alwaysAllow: string[]
  verbose: boolean
  k1: number
  b: number
}

export const DEFAULT_ALWAYS_ALLOW = ['tool_search']

export function resolveConfig(config: RouterConfig = {}): ResolvedConfig {
  const topK = config.topK ?? 8
  if (!Number.isInteger(topK) || topK <= 0) {
    throw new Error('topK must be a positive integer')
  }
  const k1 = config.k1 ?? 1.5
  const b = config.b ?? 0.75
  if (k1 <= 0 || b < 0 || b > 1) {
    throw new Error('k1 must be positive and b must be in [0, 1]')
  }

  return {
    shadow: config.shadow ?? true,
    topK,
    logPath: config.logPath,
    alwaysAllow: unique([...(config.alwaysAllow ?? DEFAULT_ALWAYS_ALLOW)]),
    verbose: config.verbose ?? true,
    k1,
    b,
  }
}

export function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))]
}
