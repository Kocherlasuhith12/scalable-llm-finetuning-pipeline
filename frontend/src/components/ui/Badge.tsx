import React from "react";

export interface BadgeProps {
  variant?: "purple" | "success" | "warning" | "error" | "neutral" | "outline";
  dot?: boolean;
  pulse?: boolean;
  children: React.ReactNode;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = "purple",
  dot = true,
  pulse = false,
  children,
  className = ""
}) => {
  const styles = {
    purple: "bg-[#8B5CF6]/15 text-[#A78BFA] border-[#8B5CF6]/30",
    success: "bg-[#22C55E]/15 text-[#22C55E] border-[#22C55E]/30",
    warning: "bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/30",
    error: "bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/30",
    neutral: "bg-[#27272A] text-[#94A3B8] border-[#3f3f46]",
    outline: "bg-transparent text-[#94A3B8] border-[#27272A]"
  };

  const dotColors = {
    purple: "bg-[#8B5CF6]",
    success: "bg-[#22C55E]",
    warning: "bg-[#F59E0B]",
    error: "bg-[#EF4444]",
    neutral: "bg-[#94A3B8]",
    outline: "bg-[#94A3B8]"
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${styles[variant]} ${className}`}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]} ${pulse ? "animate-pulse" : ""}`}
        />
      )}
      {children}
    </span>
  );
};
