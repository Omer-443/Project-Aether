'use client'

import { ReactNode } from 'react'
import { useEffect, useState } from 'react'

export function MainGrid({ children }: { children: ReactNode }) {
  const [paddingLeft, setPaddingLeft] = useState(240)

  useEffect(() => {
    let resizeObserver: ResizeObserver | null = null
    const update = () => {
      const el = document.getElementById('aether-sidebar')
      if (!el) return
      const w = el.getBoundingClientRect().width
      setPaddingLeft(Math.max(64, Math.min(240, Math.round(w))))
    }

    update()
    const el = document.getElementById('aether-sidebar')
    if (el && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(update)
      resizeObserver.observe(el)
    }
    window.addEventListener('resize', update)
    return () => {
      resizeObserver?.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [])

  return (
    <main
      className="fixed top-15 right-0 bottom-0 left-0 overflow-auto"
      style={{ paddingLeft }}
    >
      <div className="p-6 max-w-7xl mx-auto">{children}</div>
    </main>
  )
}
