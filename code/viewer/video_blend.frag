uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float weight;
uniform float highlight;

varying vec2 frag_texcoord;

void main() {
    vec4 rgba0 = texture2D(tex0, frag_texcoord);
    vec4 rgba1 = texture2D(tex1, frag_texcoord);
    gl_FragColor = (1.0-highlight) * (rgba0 * weight + rgba1 * (1.0 - weight)) +
            highlight * vec4(1.0,1.0,0.0,1.0);
}
