import React from "react";

export interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = "",
  width,
  height,
  borderRadius = "6px"
}) => {
  return (
    <div
      className={`bg-[#1F2530] animate-pulse ${className}`}
      style={{
        width: width ?? "100%",
        height: height ?? "16px",
        borderRadius: borderRadius
      }}
    />
  );
};

export const SkeletonCard: React.FC<{ className?: string }> = ({ className = "" }) => (
  <div className={`bg-[#171A21] border border-[#27272A] rounded-xl p-6 space-y-4 ${className}`}>
    <div className="flex justify-between items-center">
      <Skeleton width="40%" height="14px" />
      <Skeleton width="28px" height="28px" borderRadius="8px" />
    </div>
    <Skeleton width="60%" height="32px" />
    <Skeleton width="80%" height="12px" />
  </div>
);

export const SkeletonRow: React.FC<{ columns?: number }> = ({ columns = 4 }) => (
  <tr className="border-b border-[#27272A]/50">
    {Array.from({ length: columns }).map((_, i) => (
      <td key={i} className="py-3 px-4">
        <Skeleton width={i === 0 ? "70%" : i === 1 ? "90%" : "50%"} height="14px" />
      </td>
    ))}
  </tr>
);

export const SkeletonTable: React.FC<{ rows?: number; columns?: number }> = ({ rows = 5, columns = 4 }) => (
  <div className="w-full space-y-3">
    <div className="flex justify-between items-center pb-2">
      <Skeleton width="180px" height="32px" />
      <Skeleton width="120px" height="32px" />
    </div>
    <div className="border border-[#27272A] rounded-xl overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[#27272A] bg-[#111318]">
            {Array.from({ length: columns }).map((_, i) => (
              <th key={i} className="py-3 px-4">
                <Skeleton width="60%" height="12px" />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <SkeletonRow key={i} columns={columns} />
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export const SkeletonChart: React.FC<{ height?: string }> = ({ height = "220px" }) => (
  <div className="bg-[#171A21] border border-[#27272A] rounded-xl p-6 space-y-4">
    <div className="flex justify-between items-center">
      <Skeleton width="40%" height="16px" />
      <Skeleton width="80px" height="24px" borderRadius="12px" />
    </div>
    <div className="flex items-end gap-3 pt-4" style={{ height }}>
      {Array.from({ length: 10 }).map((_, i) => (
        <Skeleton
          key={i}
          width="100%"
          height={`${30 + Math.floor(Math.sin(i) * 30 + 40)}%`}
          borderRadius="4px"
        />
      ))}
    </div>
  </div>
);
