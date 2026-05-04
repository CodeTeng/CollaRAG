import { cn } from "@/lib/utils";

interface Props {
  className?: string;
  /** 是否同时显示"嵌合兽"主体与两创新点的三个分支 */
  showBranches?: boolean;
}

/**
 * Chimera-RAG 品牌标志
 *
 * 视觉语义：
 *   中央菱形 = 知识图谱节点（中枢），外围三条对称分叉 = 狮 / 羊 / 蛇三头。
 *   左侧金色分支 → 创新点一 AdaGraph
 *   右侧丹砂分支 → 创新点二 IntentAgent
 *   顶部青色分支 → 检索 / 问答枢纽
 */
export function ChimeraLogo({ className, showBranches = true }: Props) {
  return (
    <svg
      viewBox="0 0 64 64"
      className={cn("shrink-0", className)}
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Chimera-RAG"
    >
      <defs>
        <linearGradient id="cmr-gold" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#f5d27a" />
          <stop offset="1" stopColor="#b8872c" />
        </linearGradient>
        <linearGradient id="cmr-crimson" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#e25858" />
          <stop offset="1" stopColor="#8a2626" />
        </linearGradient>
        <linearGradient id="cmr-azure" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#7ac7dc" />
          <stop offset="1" stopColor="#2a6a7c" />
        </linearGradient>
      </defs>

      {/* 外环：图谱轨迹 */}
      <circle
        cx="32"
        cy="32"
        r="24"
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="1"
        strokeDasharray="2 3"
      />

      {/* 中央菱形：中枢节点 */}
      <g transform="translate(32 32)">
        <path
          d="M0 -10 L10 0 L0 10 L-10 0 Z"
          fill="url(#cmr-gold)"
          stroke="#c89a3a"
          strokeWidth="1.2"
        />
        <circle cx="0" cy="0" r="2.2" fill="#1a1a1a" />
      </g>

      {showBranches && (
        <>
          {/* 左分支：AdaGraph，从中心指向左下角 */}
          <g stroke="url(#cmr-gold)" strokeWidth="2" strokeLinecap="round" fill="none">
            <line x1="23" y1="36" x2="10" y2="50" />
            <circle cx="10" cy="50" r="3" fill="#d4af37" stroke="none" />
          </g>

          {/* 右分支：IntentAgent，从中心指向右下角 */}
          <g stroke="url(#cmr-crimson)" strokeWidth="2" strokeLinecap="round" fill="none">
            <line x1="41" y1="36" x2="54" y2="50" />
            <circle cx="54" cy="50" r="3" fill="#c43c3c" stroke="none" />
          </g>

          {/* 顶分支：检索枢纽 */}
          <g stroke="url(#cmr-azure)" strokeWidth="2" strokeLinecap="round" fill="none">
            <line x1="32" y1="22" x2="32" y2="8" />
            <circle cx="32" cy="8" r="3" fill="#6aa8bc" stroke="none" />
          </g>
        </>
      )}
    </svg>
  );
}
