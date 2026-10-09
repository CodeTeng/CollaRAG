// Ambient module declarations for packages without types.

declare module "react-cytoscapejs" {
  import { Core, ElementDefinition, LayoutOptions, Stylesheet } from "cytoscape";
  import { ComponentType, CSSProperties } from "react";

  interface Props {
    elements: ElementDefinition[];
    stylesheet?: Stylesheet[];
    layout?: LayoutOptions;
    style?: CSSProperties;
    className?: string;
    cy?: (cy: Core) => void;
  }

  const CytoscapeComponent: ComponentType<Props>;
  export default CytoscapeComponent;
}
